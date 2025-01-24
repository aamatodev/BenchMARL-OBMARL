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

    nodes_landmark_dist = graph_distance(node_pos, landmark_pos)
    labels = (nodes_landmark_dist < 0.5).int()

    # Compute the squared Euclidean distance between the embeddings
    distance = torch.cdist(embedding_a, embedding_b, p=2)[:, 1, 1].unsqueeze(1).unsqueeze(2).repeat(1, 3, 1)

    # Calculate the reward using the margin-based function
    reward = torch.max(torch.zeros(distance.shape).to(distance.device), margin - distance)

    return labels, distance, reward


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


class SimpleSpreadObjectiveMatchingGNN(Model):
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

        self.positive_obs_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))
        self.negative_obs_buffer = ReplayBuffer(storage=LazyTensorStorage(10000))

        self.fill_buffer(batch_size=10000)

        print(f"The positive buffer has {len(self.positive_obs_buffer)} elements")
        print(f"The negative buffer has {len(self.negative_obs_buffer)} elements")

    def fill_buffer(self, batch_size=150):
        # Fill the positive buffer
        # for positive samples, the agent and objective graph are the same
        agent_pos = (torch.rand(batch_size, 3, 2).to(device=self.device) * 2) - 1
        agent_vel = torch.zeros(batch_size, 3, 2).to(device=self.device)
        single_landmark_positions = agent_pos.view(batch_size, 1, 6)
        landmark_pos = single_landmark_positions.repeat(1, 3, 1)

        # Step 2: Reshape
        objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
        obj_shape = objective_pos.shape[0]
        obj_vel = torch.zeros(obj_shape, 2).to(device=self.device)

        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 6) - objective_pos.repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        # compute other agents positions
        relative_other_positions = relative_env_landmarks[relative_env_landmarks != 0].view(-1, 4).to(
            device=self.device)

        # Reshape the final tensor
        objective_node_features = torch.cat([objective_pos,
                                             obj_vel,
                                             relative_env_landmarks,
                                             relative_other_positions], dim=1).view(-1, 14).to(device=self.device)

        shuffled_agent = torch.empty_like(agent_pos)  # Prepare a tensor for the output
        shuffled_rel_other = torch.empty_like(
            relative_other_positions.view(batch_size, 3, -1))  # Prepare a tensor for the output
        shuffled_rel_landmarks = torch.empty_like(
            relative_env_landmarks.view(batch_size, 3, -1))  # Prepare a tensor for the output
        for i in range(batch_size):
            perm = torch.randperm(3)  # Generate a random permutation for this row
            shuffled_agent[i] = agent_pos[i, perm, :]
            shuffled_rel_other[i] = relative_other_positions.view(batch_size, 3, -1)[i, perm, :]
            shuffled_rel_landmarks[i] = relative_env_landmarks.view(batch_size, 3, -1)[i, perm, :]

        node_features = torch.cat(
            [shuffled_agent.view(-1, 2), agent_vel.view(-1, 2), shuffled_rel_landmarks.view(-1, 6),
             shuffled_rel_other.view(-1, 4)], dim=1)

        data = TensorDict(
            {
                "node_feature": node_features.view(batch_size, 3, 14),
                "node_pos": shuffled_agent.view(batch_size, 3, 2),
                "objective_node_feature": objective_node_features.view(batch_size, 3, 14),
                "objective_node_pos": objective_pos.view(batch_size, 3, 2),
            },
            batch_size=[batch_size],
        )
        self.positive_obs_buffer.extend(data)

        # Fill the negative buffer
        # for negative samples, the agent and objective graph are different
        agent_pos = (torch.rand(batch_size, 3, 2).to(device=self.device) * 2) - 1
        agent_vel = torch.zeros(batch_size, 3, 2).to(device=self.device)
        single_landmark_positions = ((torch.rand(batch_size, 3, 2) * 2) - 1).view(batch_size, 1, 6).to(
            device=self.device)
        landmark_pos = single_landmark_positions.repeat(1, 3, 1)

        # Step 2: Reshape
        objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
        obj_shape = objective_pos.shape[0]
        obj_vel = torch.zeros(obj_shape, 2).to(device=self.device)

        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 6) - objective_pos.repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        # compute other agents positions
        relative_other_positions = relative_env_landmarks[relative_env_landmarks != 0].view(-1, 4).to(
            device=self.device)

        # Reshape the final tensor
        objective_node_features = torch.cat([objective_pos,
                                             obj_vel,
                                             relative_env_landmarks,
                                             relative_other_positions], dim=1).view(-1, 14).to(device=self.device)

        # generate rel pos for nodes
        # compute the relative landmark positions
        relative_env_landmarks = landmark_pos.view(-1, 6) - agent_pos.view(-1, 2).repeat(1, landmark_pos.size(
            2) // objective_pos.size(1)).to(device=self.device)

        relative_other_positions = agent_pos.view(-1, 6).unsqueeze(1).repeat(1, 3, 1) - agent_pos.repeat(1, 1,
                                                                                                         landmark_pos.size(
                                                                                                             2) // objective_pos.size(
                                                                                                             1))

        # compute other agents positions
        relative_other_positions = relative_other_positions[relative_other_positions != 0].view(-1, 4).to(
            device=self.device)

        node_features = torch.cat(
            [agent_pos.view(-1, 2),
             agent_vel.view(-1, 2),
             relative_env_landmarks.view(-1, 6),
             relative_other_positions.view(-1, 4)], dim=1)

        data = TensorDict(
            {
                "node_feature": node_features.view(batch_size, 3, 14),
                "node_pos": agent_pos.view(batch_size, 3, 2),
                "objective_node_feature": objective_node_features.view(batch_size, 3, 14),
                "objective_node_pos": objective_pos.view(batch_size, 3, 2),
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
            landmark_positions = tensordict.get("agents")["observation"][
                "landmark_pos"]  # Shape: [batch_size, n_agents, n_agents * 2]
            # Step 1: Keep only the first element in the second dimension
            single_landmark_positions = landmark_positions[:, :1, :]  # Shape: [batch_size, 1, n_agents * 2]
            # Step 2: Reshape
            objective_pos = single_landmark_positions.reshape(-1, 2)  # Shape: [batch_size, n_agents, 2]
            obj_shape = objective_pos.shape[0]
            objective_vel = torch.zeros(obj_shape, 2).to(device=self.device)

            # compute the relative landmark positions
            relative_env_landmarks = landmark_positions.view(-1, 6) - objective_pos.repeat(1, landmark_positions.size(
                2) // objective_pos.size(1)).to(device=self.device)

            # compute other agents positions
            relative_other_positions = relative_env_landmarks[relative_env_landmarks != 0].view(-1, 4).to(
                device=self.device)

            # cat final objective node features
            objective_node_features = torch.cat(
                [objective_pos, objective_vel, relative_env_landmarks, relative_other_positions], dim=1)

            # Objective graph representation
            graphs = generate_graph(batch_size, objective_node_features, objective_pos, None, self.n_agents,
                                    self.device)
            h1 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # create the agent - agent graph
            agent_positions = tensordict.get("agents")["observation"]["agent_pos"]
            agent_velocities = tensordict.get("agents")["observation"]["agent_vel"]
            other_pos = tensordict.get("agents")["observation"]["other_pos"]
            relative_landmarks = tensordict.get("agents")["observation"]["relative_landmark_pos"]

            # agent features
            agents_features = torch.cat([agent_positions,
                                         agent_velocities,
                                         relative_landmarks,
                                         other_pos], dim=2).view(-1, 14)

            graphs = generate_graph(batch_size, agents_features, agent_positions.view(-1, 2), None,
                                    self.n_agents, self.device)

            # Agent-Agent graph representation
            h2 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # perform global pool
            objective_pooling = torch_geometric.nn.global_add_pool(h1, graphs.batch)
            agent_pooling = torch_geometric.nn.global_add_pool(h2, graphs.batch)

            labels, distance, c_rew = contrastive_reward(agent_positions,
                                                         single_landmark_positions.view(batch_size, self.n_agents, 2),
                                                         objective_pooling.unsqueeze(1).repeat(1, self.n_agents, 1),
                                                         agent_pooling.unsqueeze(1).repeat(1, self.n_agents, 1))

            # sample positive examples
            positive_data = self.positive_obs_buffer.sample(batch_size)
            positive_agent_pos = positive_data.get("node_pos").to(self.device)
            positive_agent_features = positive_data.get("node_feature").to(self.device)
            positive_objective_pos = positive_data.get("objective_node_pos").to(self.device)
            positive_objective_features = positive_data.get("objective_node_feature").to(self.device)

            # positive agent graph
            positive_agent_graph = generate_graph(batch_size, positive_agent_features.view(-1, 14),
                                                  positive_agent_pos.view(-1, 2), None, self.n_agents,
                                                  self.device)
            positive_agent_graph_embedding = F.relu(
                self.matching_gnn(x=positive_agent_graph.x, edge_index=positive_agent_graph.edge_index,
                                  edge_attr=positive_agent_graph.edge_attr))
            positive_agent_graph_pooling = torch_geometric.nn.global_add_pool(positive_agent_graph_embedding,
                                                                              positive_agent_graph.batch)

            # positive objective graph
            positive_objective_graph = generate_graph(batch_size, positive_objective_features.view(-1, 14),
                                                      positive_objective_pos.view(-1, 2), None,
                                                      self.n_agents, self.device)
            positive_objective_graph_embedding = F.relu(
                self.matching_gnn(x=positive_objective_graph.x, edge_index=positive_objective_graph.edge_index,
                                  edge_attr=positive_objective_graph.edge_attr))
            positive_objective_graph_pooling = torch_geometric.nn.global_add_pool(positive_objective_graph_embedding,
                                                                                  positive_objective_graph.batch)

            # negative examples graph
            negative_data = self.negative_obs_buffer.sample(batch_size)
            negative_agent_pos = negative_data.get("node_pos").to(self.device)
            negative_agent_features = negative_data.get("node_feature").to(self.device)
            negative_objective_pos = negative_data.get("objective_node_pos").to(self.device)
            negative_objective_features = negative_data.get("objective_node_feature").to(self.device)

            # negative agent graph
            negative_agent_graph = generate_graph(batch_size, negative_agent_features.view(-1, 14),
                                                  negative_agent_pos.view(-1, 2), None, self.n_agents,
                                                  self.device)
            negative_agent_graph_embedding = F.relu(
                self.matching_gnn(x=negative_agent_graph.x, edge_index=negative_agent_graph.edge_index,
                                  edge_attr=negative_agent_graph.edge_attr))
            negative_agent_graph_pooling = torch_geometric.nn.global_add_pool(negative_agent_graph_embedding,
                                                                              negative_agent_graph.batch)

            # negative objective graph
            negative_objective_graph = generate_graph(batch_size, negative_objective_features.view(-1, 14),
                                                      negative_objective_pos.view(-1, 2), None,
                                                      self.n_agents, self.device)
            negative_objective_graph_embedding = F.relu(
                self.matching_gnn(x=negative_objective_graph.x, edge_index=negative_objective_graph.edge_index,
                                  edge_attr=negative_objective_graph.edge_attr))
            negative_objective_graph_pooling = torch_geometric.nn.global_add_pool(negative_objective_graph_embedding,
                                                                                  negative_objective_graph.batch)

            # merge representations (pos, neg, and current)
            positive_merged_rep = torch.cat([positive_agent_graph_pooling, positive_objective_graph_pooling], dim=1)
            negative_merged_rep = torch.cat([negative_agent_graph_pooling, negative_objective_graph_pooling], dim=1)
            current_merged_rep = torch.cat([agent_pooling, objective_pooling], dim=1)
            objective_merged_rep = torch.cat([objective_pooling, objective_pooling], dim=1)

            # encode the merged representations
            positive_merged_rep_encoding = self.graphs_encoder(positive_merged_rep.unsqueeze(1).repeat(1, self.n_agents, 1))
            negative_merged_rep_encoding = self.graphs_encoder(negative_merged_rep.unsqueeze(1).repeat(1, self.n_agents, 1))
            current_merged_rep_encoding = self.graphs_encoder(current_merged_rep.unsqueeze(1).repeat(1, self.n_agents, 1))
            objective_merged_rep_encoding = self.graphs_encoder(objective_merged_rep.unsqueeze(1).repeat(1, self.n_agents, 1))

            # Concatenate the agent-objective similarity to the agent-objective graph
            agent_final_obs = torch.cat([agent_pooling.unsqueeze(1).repeat(1, self.n_agents, 1),
                                         objective_pooling.unsqueeze(1).repeat(1, self.n_agents, 1),
                                         distance,
                                         agents_features.view(-1, self.n_agents, 14)
                                         ], dim=2)

            # graphs = generate_graph(batch_size, agent_final_obs.view(-1, 263), agent_positions.view(-1, 2), None,
            #                         self.n_agents, self.device, use_radius=True, bc=1)
            # h3 = F.relu(self.agent_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            res = F.relu(self.final_mlp.forward(agent_final_obs.reshape(batch_size, self.n_agents, -1)))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], current_merged_rep_encoding)
        tensordict.set(self.out_keys[2], positive_merged_rep_encoding)
        tensordict.set(self.out_keys[3], negative_merged_rep_encoding)
        tensordict.set(self.out_keys[4], c_rew)
        tensordict.set(self.out_keys[5], distance)
        tensordict.set(self.out_keys[6], labels.unsqueeze(1).unsqueeze(2).repeat(1, self.n_agents, 1))
        return tensordict


@dataclass
class SimpleSpreadObjectiveMatchingGNNConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleSpreadObjectiveMatchingGNN
