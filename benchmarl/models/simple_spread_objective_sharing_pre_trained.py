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

from contrastive_learning.model.scl_model import SCLModel
from contrastive_learning.model.scl_model_v2 import SCLModelv2
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

    indices = torch.tensor([
        [2, 3, 4, 5],  # Keep the last 4 elements for the first element
        [0, 1, 4, 5],  # Keep the first two and last two elements for the second element
        [0, 1, 2, 3],  # Keep the first four elements for the third element
    ]).to(landmark_pos.device)
    indices = indices.unsqueeze(0).expand(landmark_pos.shape[0], -1, -1)
    # Use `gather` to apply the indexing along the last dimension
    relative_other_pos = torch.gather(relative_landmarks_pos, 2, indices)

    return objective_pos, objective_vel, relative_landmarks_pos, relative_other_pos


class SimpleSpreadObjectiveSharingPreTrained(Model):
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
        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        ) - self.n_agents * 2  # we remove the "landmark_pos" from the input features

        self.raw_feature_encoder = Encoder(self.input_features, 128).to(self.device)
        self.node_feature_encoder = Encoder(277, 128).to(self.device)

        self.agents_entity_gnn = GATv2Conv(128, 128, 2, edge_dim=3).to(self.device)
        self.agents_agents_gnn = GATv2Conv(256, 128, 2, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=515,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=torch.nn.ReLU,
            depth=3,
            num_cells=[256, 128, 32],
        )

        self.graph_encoder = SCLModelv2(self.device).to(device=self.device)
        self.graph_encoder.load_state_dict(
            torch.load("./contrastive_learning/model_full_dict_large_10eps_v2.pth"))
        self.graph_encoder.eval()

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

            with torch.no_grad():
                h_agent_graph_metric = self.graph_encoder(tensordict.get("agents")["observation"])

            # create obs for agents in objective position and objective

            obs = dict()
            obs["agent_pos"] = objective_pos.view(-1, self.n_agents, 2)
            obs["landmark_pos"] = landmark_pos
            obs["agent_vel"] = objective_vel.view(batch_size, self.n_agents, 2)
            obs["relative_landmark_pos"] = objective_relative_landmarks_pos
            obs["other_pos"] = objective_relative_other_pos

            with torch.no_grad():
                h_objective_graph_encoding = self.graph_encoder(obs)

            distance = torch.pairwise_distance(h_agent_graph_metric, h_objective_graph_encoding,
                                               keepdim=True).unsqueeze(1).repeat(1, self.n_agents, 1)

            # create agent - entity graph
            # cat one agent with the 3 entities

            agents_features = torch.cat([
                agents_pos,
                agents_vel,
                relative_landmarks_pos,
                relative_other_pos], dim=2).view(batch_size, self.n_agents, 1, -1)

            objective_features = torch.cat([
                objective_pos,
                objective_vel,
                objective_relative_landmarks_pos,
                objective_relative_other_pos], dim=2).view(batch_size, 1, self.n_agents, -1).expand(-1, self.n_agents,
                                                                                                    -1, -1)

            final_tensor = torch.cat((agents_features, objective_features), dim=2).view(batch_size * self.n_agents, 4,
                                                                                        -1)

            # agents - entity positions
            agents_pos_extended = agents_pos.view(batch_size, self.n_agents, 1, -1)
            objective_pos_extended = objective_pos.view(batch_size, 1, self.n_agents, 2).expand(-1, self.n_agents, -1,
                                                                                                -1)

            agents_entity_pos_unrolled = torch.cat([agents_pos_extended, objective_pos_extended], dim=2).view(-1, 2)

            h_agents_entity_features_unrolled = self.raw_feature_encoder.forward(
                final_tensor.view(-1, self.input_features)).view(-1, 128)

            agents_entity_graph = generate_graph(batch_size=batch_size * self.n_agents,
                                                 node_features=h_agents_entity_features_unrolled,
                                                 node_pos=agents_entity_pos_unrolled,
                                                 edge_attr=None,
                                                 n_agents=self.n_agents + 1,
                                                 device=self.device)

            h_agents_entity_graph = self.agents_entity_gnn.forward(agents_entity_graph.x,
                                                                   agents_entity_graph.edge_index,
                                                                   agents_entity_graph.edge_attr).view(
                batch_size * self.n_agents, self.n_agents + 1, -1)
            h_only_agents = h_agents_entity_graph[:, 0, :]

            # now create the graph only with the agents
            h_only_agents_unrolled = h_only_agents.view(-1, 256)
            agents_pos_unrolled = agents_pos.view(-1, 2)

            agents_graph = generate_graph(batch_size=batch_size,
                                          node_features=h_only_agents_unrolled,
                                          node_pos=agents_pos_unrolled,
                                          edge_attr=None,
                                          n_agents=self.n_agents,
                                          device=self.device)

            h_agents_graph = self.agents_agents_gnn.forward(agents_graph.x,
                                                            agents_graph.edge_index,
                                                            agents_graph.edge_attr).view(batch_size, self.n_agents, -1)

            agents_final_features = torch.cat(
                [
                    h_agent_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
                    h_objective_graph_encoding.unsqueeze(1).repeat(1, self.n_agents, 1),
                    distance,
                    agents_pos,
                    h_agents_graph
                ], dim=2)

            res = self.final_mlp(agents_final_features.view(batch_size, self.n_agents, -1))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], -distance * 0.1)
        return tensordict


@dataclass
class SimpleSpreadObjectiveSharingPreTrainedConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleSpreadObjectiveSharingPreTrained
