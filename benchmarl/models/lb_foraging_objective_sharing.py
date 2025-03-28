from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from benchmarl.models.common import Model, ModelConfig
from cmodels.lb_foraging_model_v1 import FclModelV1
from cmodels.scl_model_v2 import SCLModelv2
from cmodels.scl_model_v3 import SCLModelv3
from tensordict import TensorDictBase
from torch import nn
import torch.nn.functional as F

from torchrl.modules import MultiAgentMLP


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
            graphs.pos, batch=graphs.batch, r=0.5, loop=False
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


def extract_features_from_obs(tensor, N, M):
    # Reshape to separate environments, players, and (food + player) info
    n_envs = tensor.shape[0]

    # Reshape the last dimension to (N + M, 3) to structure as (x, y, lvl)
    tensor = tensor.view(n_envs, M, -1, 3)

    # Extract food positions (first N entries)
    food_xy = tensor[:, :, :N, :2]  # Shape: [n_envs, n_players, N, 2]

    # Extract player positions (next M entries)
    player_xy = tensor[:, :, N:N + M, :2]  # Shape: [n_envs, n_players, M, 2]

    return player_xy[:, 1, :, :], food_xy[:, 1, :, :]


def generate_objective_node_features(food_pos):
    # in lbforaging, the objective for the agent is to reach the food and dispose around it
    n_landmark = food_pos.shape[1]
    objective_agents_pos = torch.zeros((food_pos.shape[0], 4, 2)).to(device=food_pos.device)
    N = torch.tensor([1, 0]).to(device=food_pos.device)
    S = torch.tensor([-1, 0]).to(device=food_pos.device)
    E = torch.tensor([0, 1]).to(device=food_pos.device)
    W = torch.tensor([0, -1]).to(device=food_pos.device)

    for i in range(food_pos.shape[0]):
        objective_agents_pos[i, 0, :] = food_pos[i, 0, :] + N
        objective_agents_pos[i, 1, :] = food_pos[i, 0, :] + E
        objective_agents_pos[i, 2, :] = food_pos[i, 0, :] + S
        objective_agents_pos[i, 3, :] = food_pos[i, 0, :] + W

    return objective_agents_pos, food_pos


class LbForagingObjectiveSharing(Model):
    def __init__(
            self,
            activation_class: Type[nn.Module],
            threshold: float,
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
        )
        self.activation_class = activation_class
        self.threshold = threshold

        self.raw_feature_encoder = Encoder(15, 128).to(self.device)
        self.context_feature_encoder = Encoder(257, 32).to(self.device)
        self.node_feature_encoder = Encoder(277, 128).to(self.device)

        self.agents_entity_gnn = GATv2Conv(128, 128, 2, edge_dim=3).to(self.device)
        self.agents_agents_gnn = GATv2Conv(128, 128, 2, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=288,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=activation_class,
            depth=2,
            num_cells=[128, 32],
        )

        self.graph_encoder = FclModelV1(self.device).to(device=self.device)
        self.graph_encoder.load_state_dict(
            torch.load("../../../Tasks/lbforaging/state_dict_lb_foraging_model_full_v1.pth"))
        self.graph_encoder.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:

            if len(tensordict.get("player")["observation"].shape) == 2:
                raw_obs = tensordict.get("player")["observation"].unsqueeze(0)
            else:
                raw_obs = tensordict.get("player")["observation"]

            agents_pos, food_pos = extract_features_from_obs(raw_obs, 1, self.n_agents)
            batch_size = agents_pos.shape[:-2][0]

            # create objective node features
            objective_pos, obj_food_pos, = generate_objective_node_features(food_pos)

            obs = dict()
            obs["agents_pos"] = agents_pos
            obs["food_pos"] = food_pos

            with torch.no_grad():
                h_agent_graph_metric = self.graph_encoder(obs)

            obs = dict()
            obs["agents_pos"] = objective_pos
            obs["food_pos"] = obj_food_pos

            with torch.no_grad():
                h_objective_graph_metric = self.graph_encoder(obs)

            distance = torch.pairwise_distance(h_agent_graph_metric, h_objective_graph_metric,
                                               keepdim=True).unsqueeze(1).repeat(1, self.n_agents, 1)

            c_reward = torch.zeros_like(distance)  # Initialize reward tensor
            # Define a small threshold where we consider the agent "arrived"
            epsilon = 3  # Adjust based on your needs

            max_reward = 10
            alpha = 0.2

            # Compute reward as before
            c_reward[distance < self.threshold] = self.threshold - distance[distance < self.threshold]

            c_reward[distance < epsilon] = max_reward * torch.exp(
                -alpha * distance[distance < epsilon])  # Apply clamping only
            # where valid

            # Once the agent is close enough, set a stable reward
            c_reward[distance < 0.1] = 100

            # create agent - entity graph
            # cat one agent with the 3 entities

            agents_features = raw_obs

            h_agents_features_enc = self.raw_feature_encoder.forward(
                agents_features.view(-1, self.input_features))

            # now create the graph only with the agents
            h_only_agents_unrolled = h_agents_features_enc.view(-1, 128)
            agents_pos_unrolled = agents_pos.reshape(-1, 2)

            agents_graph = generate_graph(batch_size=batch_size,
                                          node_features=h_only_agents_unrolled,
                                          node_pos=agents_pos_unrolled,
                                          edge_attr=None,
                                          n_agents=self.n_agents,
                                          use_radius=True,
                                          device=self.device)

            h_agents_graph = self.agents_agents_gnn.forward(agents_graph.x,
                                                            agents_graph.edge_index,
                                                            agents_graph.edge_attr).view(batch_size, self.n_agents, -1)
            agents_id = torch.arange(self.n_agents, device=self.device).unsqueeze(0).expand(batch_size, -1).unsqueeze(2)

            context_features = torch.cat(
                [
                    h_agent_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
                    h_objective_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
                    distance,
                ], dim=2)

            context_encoded = self.context_feature_encoder.forward(context_features.view(-1, 257)).view(batch_size,
                                                                                                        self.n_agents,
                                                                                                        -1)

            agents_final_features = torch.cat(
                [
                    context_encoded,
                    h_agents_graph
                ], dim=2)

            res = self.final_mlp(agents_final_features.view(batch_size, self.n_agents, -1))

        if len(tensordict.get("player")["observation"].shape) == 2:
            tensordict.set(self.out_keys[0], res.squeeze(0))
            tensordict.set(self.out_keys[1], c_reward.squeeze(0))
        else:
            tensordict.set(self.out_keys[0], res)
            tensordict.set(self.out_keys[1], c_reward)

        return tensordict


@dataclass
class LbForagingObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    threshold: float = 50.0

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return LbForagingObjectiveSharing
