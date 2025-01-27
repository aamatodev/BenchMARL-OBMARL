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
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def graph_distance(objective_node_features, agent_node_features):
    graph_dist = []
    for i in range(objective_node_features.shape[0]):
        env_dist = []
        for j in range(objective_node_features.shape[1]):
            distance = torch.min(torch.linalg.norm(objective_node_features[i][j] - agent_node_features[i], dim=1))
            env_dist.append(distance)
        graph_dist.append(torch.sum(torch.stack(env_dist)))
    return torch.stack(graph_dist)


def contrastive_reward(node_pos, landmark_pos, embedding_a, embedding_b, margin=2):
    """
    Calculate the reward based on contrastive loss-inspired function.

    Args:
        embedding_a (np.ndarray): The first embedding (agent's current state).
        embedding_b (np.ndarray): The target embedding.
        margin (float): The margin threshold for determining the reward.

    Returns:
        float: The reward value.
    """

    # Compute the squared Euclidean distance between the embeddings
    distance = torch.cdist(embedding_a, embedding_b, p=2)[:, 1].unsqueeze(1).unsqueeze(2).repeat(1, 3, 1)

    # Calculate the reward using the margin-based function
    reward = torch.max(torch.zeros(distance.shape).to(distance.device), margin - distance)

    return distance, reward


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


class SimpleSpreadObjectiveMatchingGNNPreTrained(Model):
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
        self.matching_gnn = GATv2Conv(14, 32, 1, edge_dim=3).to(self.device)

        self.graphs_encoder = MultiAgentMLP(
            n_agent_inputs=64,
            n_agent_outputs=32,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=True,
            device=self.device,
            activation_class=self.activation_function,
            depth=1,
            num_cells=[128],
        )

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=79,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_function,
            depth=2,
            num_cells=[32, 32],
        )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            batch_size = tensordict.get("agents")["observation"]["agent_pos"].shape[:-2][0]
            # load the pre-trained model
            graph_encoder = SCLModel(self.device).to(device=self.device)
            graph_encoder.load_state_dict(
                torch.load("/home/aamato/Documents/marl/objective-based-marl/contrastive_learning/model_full_dict.pth"))
            graph_encoder.eval()

            agent_graph_encoding = graph_encoder(tensordict.get("agents")["observation"])

            # set the agent pos to objective pos for the objective graph

            # reorganize the landmark positions for the matching GNN
            landmark_positions = tensordict.get("agents")["observation"][
                "landmark_pos"]  # Shape: [batch_size, n_agents, n_agents * 2]
            # Step 1: Keep only the first element in the second dimension
            single_landmark_positions = landmark_positions[:, :1, :]  # Shape: [batch_size, 1, n_agents * 2]
            # Step 2: Reshape
            objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
            obj_shape = objective_pos.shape[0]
            objective_vel = torch.zeros(obj_shape, 2).to(device=self.device)

            # compute the relative landmark positions
            relative_env_landmarks = (landmark_positions.view(-1, 6) - objective_pos.repeat(1, landmark_positions.size(
                2) // objective_pos.size(1))).view(batch_size, 3, 6).to(device=self.device)

            # compute other agents positions
            relative_other_positions = relative_env_landmarks[relative_env_landmarks != 0].view(-1, 4).to(
                device=self.device)

            obs = dict()
            obs["agent_pos"] = landmark_positions[:, :1, :].view(-1, 3, 2)
            obs["landmark_pos"] = tensordict.get("agents")["observation"]["landmark_pos"]

            obs["agent_vel"] = tensordict.get("agents")["observation"]["agent_vel"]
            obs["relative_landmark_pos"] = relative_env_landmarks
            obs["other_pos"] = relative_other_positions.view(batch_size, 3, 4)

            objective_graph_encoding = graph_encoder(obs)

            distance, c_rew = contrastive_reward([],
                                                         single_landmark_positions.view(batch_size, self.n_agents, 2),
                                                         agent_graph_encoding,
                                                         objective_graph_encoding)


            agent_positions = tensordict.get("agents")["observation"]["agent_pos"]
            agent_velocities = tensordict.get("agents")["observation"]["agent_vel"]
            other_pos = tensordict.get("agents")["observation"]["other_pos"]
            relative_landmarks = tensordict.get("agents")["observation"]["relative_landmark_pos"]

            # agent features
            agents_features = torch.cat([agent_positions,
                                         agent_velocities,
                                         relative_landmarks,
                                         other_pos], dim=2).view(-1, 14)


            # Concatenate the agent-objective similarity to the agent-objective graph
            agent_final_obs = torch.cat([agent_graph_encoding.unsqueeze(1).repeat(1, self.n_agents, 1),
                                         objective_graph_encoding.unsqueeze(1).repeat(1, self.n_agents, 1),
                                         distance,
                                         agents_features.view(-1, self.n_agents, 14)
                                         ], dim=2)

            res = F.relu(self.final_mlp.forward(agent_final_obs.reshape(batch_size, self.n_agents, -1)))

        tensordict.set(self.out_keys[0], res)
        # tensordict.set(self.out_keys[1], agent_graph_encoding)
        # tensordict.set(self.out_keys[2], objective_graph_encoding)
        # tensordict.set(self.out_keys[3], c_rew)
        # tensordict.set(self.out_keys[4], distance)
        # tensordict.set(self.out_keys[5], labels.unsqueeze(1).unsqueeze(2).repeat(1, self.n_agents, 1))

        return tensordict


@dataclass
class SimpleSpreadObjectiveMatchingGNNPreTrainedConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleSpreadObjectiveMatchingGNNPreTrained
