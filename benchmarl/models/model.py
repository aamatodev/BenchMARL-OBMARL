import numpy as np
import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import GATv2Conv, Linear
import torch.nn.functional as F


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(torch.nn.Module):
    """Encoder for initial state of DGN"""

    def __init__(self, num_input_feature, num_output_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
    def forward(self, obs):
        embedding = self.fc(obs)
        return embedding


class SCLModel(torch.nn.Module):
    def __init__(self, device):
        super(SCLModel, self).__init__()

        self.matching_gnn = GATv2Conv(in_channels=14, out_channels=16, heads=1, edge_dim=3)
        self.encoder = Encoder(num_input_feature=32, num_output_feature=32)
        self.device = device
        self.n_agents = 3

    def forward(self, x):
        # create the objective graph
        batch_size = x["agent_pos"].shape[:-2][0]

        # reorganize the landmark positions for the matching GNN
        landmark_positions = x["landmark_pos"]  # Shape: [batch_size, n_agents, n_agents * 2]
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
        agent_positions = x["agent_pos"]
        agent_velocities = x["agent_vel"]
        other_pos = x["other_pos"]
        relative_landmarks = x["relative_landmark_pos"]

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

        h = torch.cat([agent_pooling, objective_pooling], dim=1)

        final_emb = self.encoder(h)

        return final_emb
