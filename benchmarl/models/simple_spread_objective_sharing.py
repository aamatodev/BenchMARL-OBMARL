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

from torchrl.data import Composite, Unbounded, ReplayBuffer, LazyTensorStorage

from contrastive_learning.model.model import SCLModel
from tensordict import TensorDictBase, TensorDict
from torch import nn, cosine_similarity
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


def extract_features_from_obs(obs):
    agents_pos = obs["agent_pos"]
    agents_vel = obs["agent_vel"]
    landmarks_pos = obs["landmark_pos"]
    relative_landmarks_pos = obs["relative_landmark_pos"]
    relative_other_pos = obs["other_pos"]

    return agents_pos, agents_vel, landmarks_pos, relative_landmarks_pos, relative_other_pos


def generate_objective_node_features(landmark_pos):
    # in simple spread, the objective is reached once all the landmarks are covered. This happens when the agents
    # positions are equals to the landmarks positions
    n_landmark = landmark_pos.shape[1]
    objective_pos = landmark_pos[:, 1, :].view(-1, n_landmark, 2).clone().detach()
    objective_vel = torch.zeros_like(objective_pos)

    relative_landmarks_pos = landmark_pos - objective_pos.repeat(1, 1, n_landmark)

    relative_other_pos = relative_landmarks_pos[relative_landmarks_pos != 0].view(-1, n_landmark, (n_landmark - 1) * 2)

    return objective_pos, objective_vel, relative_landmarks_pos, relative_other_pos


class SimpleSpreadObjectiveSharing(Model):
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

        self.node_feature_encoder_1 = Encoder(14, 128).to(self.device)
        self.node_feature_encoder_2 = Encoder(512*3, 512).to(self.device)
        self.matching_gnn = GATv2Conv(128, 128, 4, edge_dim=3).to(self.device)
        self.matching_gnn_2 = GATv2Conv(512, 128, 4, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=512,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=torch.nn.ReLU,
            depth=3,
            num_cells=[128, 128, 32],
        )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            agents_pos, agents_vel, landmark_pos, relative_landmarks_pos, relative_other_pos = extract_features_from_obs(
                tensordict.get("agents")["observation"])
            batch_size = agents_pos.shape[:-2][0]

            # create objective node features
            objective_pos, objective_vel, objective_relative_landmarks_pos, objective_relative_other_pos = generate_objective_node_features(
                landmark_pos)

            # create graph context for both objective and agents graph
            current_agents_features_unrolled = torch.cat(
                [agents_pos, agents_vel, relative_landmarks_pos, relative_other_pos], dim=-1).view(
                batch_size * self.n_agents, -1)
            objective_node_features_unrolled = torch.cat(
                [objective_pos, objective_vel, objective_relative_landmarks_pos, objective_relative_other_pos],
                dim=-1).view(batch_size * self.n_agents, -1)

            # encode agents features
            h_current_agents = self.node_feature_encoder_1.forward(current_agents_features_unrolled)
            h_objective = self.node_feature_encoder_1.forward(objective_node_features_unrolled)

            # create graphs
            agents_graph = generate_graph(batch_size=batch_size,
                                          node_features=h_current_agents,
                                          node_pos=agents_pos.view(batch_size * self.n_agents, -1),
                                          edge_attr=None,
                                          n_agents=self.n_agents,
                                          device=self.device)

            objective_graph = generate_graph(batch_size=batch_size,
                                             node_features=h_objective,
                                             node_pos=objective_pos.view(batch_size * self.n_agents, -1),
                                             edge_attr=None,
                                             n_agents=self.n_agents,
                                             device=self.device)

            # encode graphs
            h_agents_graph = self.matching_gnn.forward(agents_graph.x, agents_graph.edge_index, agents_graph.edge_attr)
            h_objective_graph = self.matching_gnn.forward(objective_graph.x, objective_graph.edge_index,
                                                          objective_graph.edge_attr)

            # aggregate graphs with pooling
            h_agents_graph_add_pooling = torch_geometric.nn.global_add_pool(h_agents_graph, agents_graph.batch)
            h_objective_graph_add_pooling = torch_geometric.nn.global_add_pool(h_objective_graph,
                                                                                objective_graph.batch)

            # create agents - objective node features
            agents_objective = torch.cat(
                [h_agents_graph.view(batch_size, self.n_agents, -1),
                 h_objective_graph.view(batch_size, self.n_agents, -1)], dim=1)

            agents_objective_pos = torch.cat([agents_pos, objective_pos], dim=1)

            agents_objective_features = torch.cat(
                [h_agents_graph_add_pooling.unsqueeze(1).repeat(1, 6, 1),
                 h_objective_graph_add_pooling.unsqueeze(1).repeat(1, 6, 1),
                 agents_objective], dim=2)

            # encode agents - objective node features
            h_agents_objective_unrolled = self.node_feature_encoder_2.forward(agents_objective_features).view(batch_size * self.n_agents * 2, -1)

            # generate final graph
            final_graph = generate_graph(batch_size=batch_size,
                                         node_features=h_agents_objective_unrolled,
                                         node_pos=agents_objective_pos.view(batch_size * self.n_agents * 2, -1),
                                         edge_attr=None,
                                         n_agents=self.n_agents * 2,
                                         device=self.device)

            h_agents_objective_graph = self.matching_gnn_2.forward(final_graph.x,
                                                                   final_graph.edge_index,
                                                                   final_graph.edge_attr).view(batch_size, self.n_agents * 2, -1)[:, :3, :]

            res = self.final_mlp.forward(h_agents_objective_graph)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class SimpleSpreadObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleSpreadObjectiveSharing
