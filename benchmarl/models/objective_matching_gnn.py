from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

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


class ObjectiveMatchingGNN(Model):
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

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )

        self.output_features = self.output_leaf_spec.shape[-1]

        self.activation_function = activation_class

        self.node_pos_encoder = Encoder(2, 8).to(self.device)
        self.edge_encoder = Encoder(2, 8).to(self.device)
        self.matching_gnn = GATv2Conv(2, 128, 1, edge_dim=3).to(self.device)
        self.agent_gnn = GATv2Conv(269, 128, 1, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=128,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_function,
            depth=3,
            num_cells=[128, 64, 32],
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
            landmark_positions = landmark_positions[:, :1, :]  # Shape: [10, 1, 10]
            # Step 2: Remove the first two elements in the third dimension
            landmark_positions = landmark_positions[:, :, 2:]  # Shape: [10, 1, 8]
            # Step 3: Reshape to [10, 4, 2]

            landmark_positions = landmark_positions.reshape(-1, 2)

            # Commenting this out given that we don't want it in the initial objetive graph
            # [:, :, 1:] to remove the first column which is the agent status
            # landmark_eaten = tensordict.get("agents")["observation"]["landmark_eaten"][:, :, 1:]
            # landmark_eaten = landmark_eaten.view(-1, 1)
            # node_features = torch.cat([landmark_positions, landmark_eaten], dim=1)

            # Objective graph representation
            graphs = generate_graph(batch_size, landmark_positions, landmark_positions, None, self.n_agents,
                                    self.device)
            h1 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # create the agent - agent graph
            agent_positions = tensordict.get("agents")["observation"]["agent_pos"]

            graphs = generate_graph(batch_size, agent_positions.view(-1, 2), agent_positions.view(-1, 2), None,
                                    self.n_agents, self.device)
            # Agent-Agent graph representation
            h2 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # Get cosine similarity between agent and objective graph
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            agent_objective_similarity = cos(h1, h2)

            # Concatenate the agent-objective similarity to the agent-objective graph
            agent_final_obs = torch.cat([h1.reshape(batch_size, self.n_agents, -1),
                                         h2.reshape(batch_size, self.n_agents, -1),
                                         agent_objective_similarity.reshape(batch_size, self.n_agents, -1),
                                         tensordict.get("agents")["observation"]["relative_landmark_pos"]], dim=2)

            graphs = generate_graph(batch_size, agent_final_obs.view(-1, 269), agent_positions.view(-1, 2), None,
                                    self.n_agents, self.device, use_radius=True, bc=1)
            h3 = F.relu(self.agent_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            res = F.relu(self.final_mlp.forward(h3.reshape(batch_size, self.n_agents, -1)))

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class ObjectiveMatchingGNNConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return ObjectiveMatchingGNN
