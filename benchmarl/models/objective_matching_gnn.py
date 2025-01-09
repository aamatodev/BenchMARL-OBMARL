from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torchrl.data import Composite, Unbounded, ReplayBuffer, LazyTensorStorage

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



def contrastive_reward(embedding_a, embedding_b, margin=10):
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
    distance = torch.linalg.norm(embedding_a - embedding_b, dim=-1) ** 2

    # Calculate the reward using the margin-based function
    reward = torch.max(torch.zeros(distance.shape).to(distance.device), margin - distance).unsqueeze(2)

    return distance.unsqueeze(2), reward


def graph_distance(objective_node_features, agent_node_features):
    graph_dist = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(objective_node_features.shape[0]):
        env_dist = []
        for j in range(objective_node_features.shape[1]):
            # distance = torch.min((torch.linalg.norm(objective_node_features[i][j] - agent_node_features[i], dim=1)))

            agent_objective_similarity = torch.max(cos(objective_node_features[i][j], agent_node_features[i]))
            env_dist.append(agent_objective_similarity)

        graph_dist.append((torch.sum(torch.stack(env_dist)) / 4))
    return torch.stack(graph_dist)


def generate_graph(batch_size, node_features, node_pos, edge_attr, n_agents, device, mode="complete", bc=1):
    b = torch.arange(batch_size * bc, device=device)
    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size * bc + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)

    graphs.x = node_features
    graphs.pos = node_pos
    graphs.edge_attr = edge_attr

    if mode == "radius":
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=0.5, loop=True
        )
    elif mode == "complete":
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
    elif mode == "first_node":
        adjacency = torch.zeros(n_agents, n_agents, device=device, dtype=torch.long)
        adjacency[0, :] = 1
        adjacency[:, 0] = 1
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
        self.graphs_encoder = MultiAgentMLP(
            n_agent_inputs=64,
            n_agent_outputs=32,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=True,
            device=self.device,
            activation_class=self.activation_function,
            depth=3,
            num_cells=[128, 64, 32],
        )

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=114,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_function,
            depth=3,
            num_cells=[128, 64, 32],
        )

        self.positive_obs_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))
        self.negative_obs_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))
        print(f"The positive buffer has {len(self.positive_obs_buffer)} elements")
        print(f"The negative buffer has {len(self.negative_obs_buffer)} elements")

        self.fill_buffer(batch_size=10000)

        print(f"The positive buffer has {len(self.positive_obs_buffer)} elements")
        print(f"The negative buffer has {len(self.negative_obs_buffer)} elements")

    def fill_buffer(self, batch_size=150):
        # Fill the positive buffer
        # for positive samples, the agent and objective graph are the same
        agent_pos = (torch.rand(batch_size, 4, 2).to(device=self.device) * 2) - 1
        agent_vel = torch.zeros(batch_size, 4, 2).to(device=self.device)
        single_landmark_positions = agent_pos.view(batch_size, 1, 8)
        landmark_pos = single_landmark_positions.repeat(1, 4, 1)

        # Step 2: Reshape
        objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
        obj_shape = objective_pos.shape[0]
        obj_vel = torch.zeros(obj_shape, 2).to(device=self.device)

        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 8) - objective_pos.repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        # Final tensor size: [240, 12]
        final_tensor = torch.zeros(obj_shape, 12).to(device=self.device)

        # inject the 1s as "eaten" landmarks
        positions = torch.tensor([2, 5, 8, 11]).to(device=self.device)

        # Create a mask to identify non-insert positions
        mask = torch.ones(12, dtype=torch.bool).to(device=self.device)
        mask[positions] = False

        # Place original values in the appropriate positions
        final_tensor[:, mask] = relative_env_landmarks

        # Insert 1s at the specified positions
        final_tensor[:, positions] = 1

        # Reshape the final tensor
        objective_node_features = torch.cat([objective_pos,
                                             obj_vel,
                                             final_tensor], dim=1).view(-1, 16).to(device=self.device)

        shuffled_agent = torch.empty_like(agent_pos)  # Prepare a tensor for the output
        shuffled_rel = torch.empty_like(final_tensor.view(batch_size, 4, -1))  # Prepare a tensor for the output
        for i in range(batch_size):
            perm = torch.randperm(4)  # Generate a random permutation for this row
            shuffled_agent[i] = agent_pos[i, perm, :]
            shuffled_rel[i] = final_tensor.view(batch_size, 4, -1)[i, perm, :]

        node_features = torch.cat(
            [shuffled_agent.view(-1, 2), agent_vel.view(-1, 2), shuffled_rel.view(-1, 12)], dim=1)

        data = TensorDict(
            {
                "node_feature": node_features.view(batch_size, 4, 16),
                "node_pos": shuffled_agent.view(batch_size, 4, 2),
                "objective_node_feature": objective_node_features.view(batch_size, 4, 16),
                "objective_node_pos": objective_pos.view(batch_size, 4, 2),
            },
            batch_size=[batch_size],
        )
        self.positive_obs_buffer.extend(data)

        # Fill the negative buffer
        # for negative samples, the agent and objective graph are different
        agent_pos = (torch.rand(batch_size, 4, 2).to(device=self.device) * 2) - 1
        agent_vel = torch.zeros(batch_size, 4, 2).to(device=self.device)
        single_landmark_positions = ((torch.rand(batch_size, 4, 2) * 2) - 1).view(batch_size, 1, 8).to(
            device=self.device)
        landmark_pos = single_landmark_positions.repeat(1, 4, 1)

        # Step 2: Reshape
        objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
        obj_shape = objective_pos.shape[0]
        obj_vel = torch.zeros(obj_shape, 2).to(device=self.device)

        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 8) - objective_pos.repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        # Final tensor size: [240, 12]
        final_tensor = torch.zeros(obj_shape, 12).to(device=self.device)

        # inject the 1s as "eaten" landmarks
        positions = torch.tensor([2, 5, 8, 11]).to(device=self.device)

        # Create a mask to identify non-insert positions
        mask = torch.ones(12, dtype=torch.bool).to(device=self.device)
        mask[positions] = False

        # Place original values in the appropriate positions
        final_tensor[:, mask] = relative_env_landmarks

        # Insert 1s at the specified positions
        final_tensor[:, positions] = 1

        # Reshape the final tensor
        objective_node_features = torch.cat([objective_pos,
                                             obj_vel,
                                             final_tensor], dim=1).view(-1, 16).to(device=self.device)

        # generate rel pos for nodes
        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 8) - agent_pos.view(-1, 2).repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        # Final tensor size: [240, 12]
        final_tensor = torch.zeros(obj_shape, 12).to(device=self.device)

        # inject the 1s as "eaten" landmarks
        positions = torch.tensor([2, 5, 8, 11]).to(device=self.device)

        # Create a mask to identify non-insert positions
        mask = torch.ones(12, dtype=torch.bool).to(device=self.device)
        mask[positions] = False

        # Place original values in the appropriate positions
        final_tensor[:, mask] = relative_env_landmarks

        # Insert 1s at the specified positions
        final_tensor[:, positions] = 0

        node_features = torch.cat(
            [agent_pos.view(-1, 2), agent_vel.view(-1, 2), final_tensor.view(-1, 12)], dim=1)

        data = TensorDict(
            {
                "node_feature": node_features.view(batch_size, 4, 16),
                "node_pos": agent_pos.view(batch_size, 4, 2),
                "objective_node_feature": objective_node_features.view(batch_size, 4, 16),
                "objective_node_pos": objective_pos.view(batch_size, 4, 2),
            },
            batch_size=[batch_size],
        )

        self.negative_obs_buffer.extend(data)

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
            final_tensor[:, positions] = 1

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
            h1_p = torch_geometric.nn.global_add_pool(h1, graphs.batch)
            h2 = torch_geometric.nn.global_add_pool(h2, graphs.batch)

            # Get cosine similarity between agent and objective graph
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            agent_objective_similarity = cos(h1_p, h2)

            # # sample from positive and negative buffers
            # positive_sample = self.positive_obs_buffer.sample(batch_size).to(device=self.device)
            # negative_sample = self.negative_obs_buffer.sample(batch_size).to(device=self.device)
            #
            # graphs = generate_graph(batch_size, positive_sample["node_feature"].view(-1, 16),
            #                         positive_sample["node_pos"].view(-1, 2), None,
            #                         self.n_agents, self.device)
            #
            # # Agent-Agent graph representation
            # positive_agent = F.relu(
            #     self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))
            #
            # graphs = generate_graph(batch_size, positive_sample["objective_node_feature"].view(-1, 16),
            #                         positive_sample["objective_node_pos"].view(-1, 2), None,
            #                         self.n_agents, self.device)
            # # Agent-Agent graph representation
            # positive_obj = F.relu(
            #     self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))
            #
            # graphs = generate_graph(batch_size, negative_sample["node_feature"].view(-1, 16),
            #                         negative_sample["node_pos"].view(-1, 2), None,
            #                         self.n_agents, self.device)
            #
            # # Agent-Agent graph representation
            # negative_agent = F.relu(
            #     self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))
            #
            # graphs = generate_graph(batch_size, negative_sample["objective_node_feature"].view(-1, 16),
            #                         negative_sample["objective_node_pos"].view(-1, 2), None,
            #                         self.n_agents, self.device)
            # # Agent-Agent graph representation
            # negative_obj = F.relu(
            #     self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))
            #
            # # graph pooling
            # positive_agent = torch_geometric.nn.global_add_pool(positive_agent, graphs.batch)
            # positive_obj = torch_geometric.nn.global_add_pool(positive_obj, graphs.batch)
            # negative_agent = torch_geometric.nn.global_add_pool(negative_agent, graphs.batch)
            # negative_obj = torch_geometric.nn.global_add_pool(negative_obj, graphs.batch)
            #
            # # get the final encoding
            # current_encoding_data = torch.cat([h1, h2], dim=1)
            # positive_encoding_data = torch.cat([positive_agent, positive_obj], dim=1)
            # negative_encoding_data = torch.cat([negative_agent, negative_obj], dim=1)

            # current_encoding = F.relu(
            #     self.graphs_encoder.forward(current_encoding_data.unsqueeze(1).repeat(1, 4, 1)).to(self.device))
            # positive_encoding = F.relu(
            #     self.graphs_encoder.forward(positive_encoding_data.unsqueeze(1).repeat(1, 4, 1)).to(self.device))
            # negative_encoding = F.relu(
            #     self.graphs_encoder.forward(negative_encoding_data.unsqueeze(1).repeat(1, 4, 1)).to(self.device))

            # similarity = torch.linalg.norm(h1_p - h2, dim=-1)**2

            # graphs = generate_graph(batch_size, agent_entity.view(batch_size * self.n_agents, -1, 8)[:, 1, :],
            # agent_positions.view(-1, 2), None, self.n_agents, self.device)

            # agent_agent = F.relu(self.agent_agent_gnn(x=graphs.x, edge_index=graphs.edge_index,
            # edge_attr=graphs.edge_attr))

            # similarity = graph_distance(landmark_positions.view((batch_size, self.n_agents, -1)),
            #                             agent_positions.view((batch_size, self.n_agents, -1)))

            # Concatenate the agent-objective similarity to the agent-objective graph

            distance, c_rew = contrastive_reward(h1_p.unsqueeze(1).repeat(1, 4, 1), h2.unsqueeze(1).repeat(1, 4, 1))

            agent_final_obs = torch.cat([
                h1.view(batch_size, 4, -1),
                h1_p.unsqueeze(1).repeat(1, 4, 1),
                h2.unsqueeze(1).repeat(1, 4, 1),
                # tensordict.get("agents")["observation"]["agent_index"],
                agent_objective_similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1),
                distance,
                agent_positions,
                agent_vel,
                tensordict.get("agents")["observation"]["relative_landmark_pos"]], dim=2)

            # similarity_module_obs = torch.cat([
            #     h1.unsqueeze(1).repeat(1, 4, 1),
            #     h2.unsqueeze(1).repeat(1, 4, 1),
            #     similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1),
            #     agent_objective_similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1),
            #     agent_agent.view(batch_size, 4, -1)], dim=2)

            res = F.relu(self.final_mlp.forward(agent_final_obs))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], agent_objective_similarity.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1))
        tensordict.set(self.out_keys[2], c_rew)
        tensordict.set(self.out_keys[3], distance)
        # tensordict.set(self.out_keys[3], h2.unsqueeze(1).repeat(1, 4, 1))

        return tensordict


@dataclass
class DisperseObjectiveMatchingGNNConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return DisperseObjectiveMatchingGNN
