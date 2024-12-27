from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torchrl.data import Composite, Unbounded

from benchmarl.models import Gnn, GnnConfig, DeepsetsConfig, Deepsets
from benchmarl.models.common import Model, ModelConfig
from tensordict import TensorDictBase, TensorDict
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def graph_distance(objective_node_features, agent_node_features):
    graph_dist = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(objective_node_features.shape[0]):
        env_dist = []
        for j in range(objective_node_features.shape[1]):
            # distance = torch.min((torch.linalg.norm(objective_node_features[i][j] - agent_node_features[i], dim=1))) / 8

            agent_objective_similarity = torch.max(cos(objective_node_features[i][j], agent_node_features[i]))
            env_dist.append(agent_objective_similarity)

        graph_dist.append((torch.sum(torch.stack(env_dist)) / 4))
    return torch.stack( graph_dist)


def generate_graph(batch_size, node_features, node_pos, edge_attr, n_agents, device, use_radius=False, bc=1):
    b = torch.arange(batch_size * bc, device=device)
    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size * bc + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)

    graphs.x = node_features
    graphs.pos = node_pos
    graphs.edge_attr = edge_attr

    if use_radius:
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=0.5, loop=True
        )
    else:
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        # remove self loops
        graphs.edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size * bc) + batch * n_agents
        graphs.edge_index = batch_edge_index

    graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
    graphs = torch_geometric.transforms.Distance(norm=False)(graphs)

    return graphs


class Encoder(nn.Module):
    """Encoder for initial state of DGN"""

    def __init__(self, num_input_feature, num_output_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = layer_init(nn.Linear(num_input_feature, num_output_feature))

    def forward(self, obs):
        embedding = F.relu(self.l1(obs))
        return embedding


class DisperseObjectiveMatchingGNN(Model):
    def __init__(
            self,
            activation_class: Type[nn.Module],
            **kwargs,
    ):
        # Remember the kwargs to the super() class
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        self.activation_function = activation_class

        self.node_pos_encoder = Encoder(2, 8).to(self.device)
        self.edge_encoder = Encoder(2, 8).to(self.device)
        self.matching_gnn = GATv2Conv(16, 32, 1, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=82,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_function,
            depth=2,
            num_cells=[128, 128],
        )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            # create the objective graph
            batch_size = tensordict.get("agents")["observation"]["agent_pos"].shape[:-2][0]

            # reorganize the landmark positions for the matching GNN
            landmark_positions = tensordict.get("agents")["observation"]["landmark_pos"]
            # Step 1: Keep only the first element in the second dimension
            single_landmark_positions = landmark_positions[:, :1, :]  # Shape: [batch_size, 1, n_agents * 2]
            # Step 2: Reshape
            objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
            obj_shape = objective_pos.shape[0]
            obj_vel = torch.zeros(obj_shape, 2).to(device=self.device)

            # compute the relative landmark positions
            relative_env_landmarks = landmark_positions.view(-1, 8) - objective_pos.repeat(1, landmark_positions.size(
                2) // objective_pos.size(1)).to(device=self.device)

            # Final tensor size: [240, 12]
            final_tensor = torch.zeros(obj_shape, 12).to(device=self.device)

            # Desired positions to insert 1s (0-based indexing)
            positions = torch.tensor([2, 5, 8, 11]).to(device=self.device)

            # Create a mask to identify non-insert positions
            mask = torch.ones(12, dtype=torch.bool).to(device=self.device)
            mask[positions] = False

            # Place original values in the appropriate positions
            final_tensor[:, mask] = relative_env_landmarks

            # Insert 1s at the specified positions
            final_tensor[:, positions] = 0

            # Reshape the final tensor
            objective_node_features = torch.cat([objective_pos,
                                                 obj_vel,
                                                 final_tensor], dim=1).view(-1, 16).to(device=self.device)

            # reorganize the landmark positions for the matching GNN
            landmark_positions = tensordict.get("agents")["observation"]["landmark_pos"]
            # Keep only the first element in the second dimension
            landmark_positions = landmark_positions[:, :1, :]  # Shape: [10, 1, 10]

            # Objective graph representation
            landmark_positions = landmark_positions.reshape(-1, 2)
            graphs = generate_graph(batch_size, objective_node_features, landmark_positions, None, self.n_agents,
                                    self.device)

            h1 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # create the agent - agent graph
            agent_positions = tensordict.get("agents")["observation"]["agent_pos"]
            agent_vel = tensordict.get("agents")["observation"]["agent_vel"]

            node_features = torch.cat(
                [agent_positions, agent_vel, tensordict.get("agents")["observation"]["relative_landmark_pos"]], dim=2)

            graphs = generate_graph(batch_size, node_features.view(-1, 16), agent_positions.view(-1, 2), None,
                                    self.n_agents, self.device)

            # Agent-Agent graph representation
            h2 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # graph pooling
            h1 = torch_geometric.nn.global_add_pool(h1, graphs.batch)
            h2 = torch_geometric.nn.global_add_pool(h2, graphs.batch)

            # Get cosine similarity between agent and objective graph
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            agent_objective_similarity = cos(h1, h2)

            similarity = graph_distance(objective_node_features.view((batch_size, self.n_agents, -1)),
                                        node_features.view((batch_size, self.n_agents, -1)))
            # similarity = graph_distance(landmark_positions.view((batch_size, self.n_agents, -1)),
            #                             agent_positions.view((batch_size, self.n_agents, -1)))

            # Concatenate the agent-objective similarity to the agent-objective graph
            agent_final_obs = torch.cat([
                h1.unsqueeze(1).repeat(1, 4, 1),
                h2.unsqueeze(1).repeat(1, 4, 1),
                # similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1),
                agent_objective_similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1),
                agent_positions,
                agent_vel,
                tensordict.get("agents")["observation"]["relative_landmark_pos"]], dim=2)

            res = F.relu(self.final_mlp.forward(agent_final_obs))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], h1.unsqueeze(1).repeat(1, 4, 1))
        tensordict.set(self.out_keys[2], h2.unsqueeze(1).repeat(1, 4, 1))
        tensordict.set(self.out_keys[3], similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1))

        return tensordict


@dataclass
class DisperseObjectiveMatchingGNNConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return DisperseObjectiveMatchingGNN
