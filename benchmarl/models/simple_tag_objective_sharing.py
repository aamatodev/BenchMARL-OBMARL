from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv

from Tasks.SimpleTag.contrastive_model.model.tag_contrastive_model import TagContrastiveModel
from benchmarl.models.common import Model, ModelConfig

from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MultiAgentMLP

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


def extract_features_from_obs(obs):
    agents_pos = obs["agent_pos"]
    agents_vel = obs["agent_vel"]
    entity_pos = obs["entity_pos"]
    other_pos = obs["other_pos"]
    other_vel = obs["other_vel"]

    return agents_pos, agents_vel, other_pos, other_vel, entity_pos


def generate_objective_state_predators(pray_pos, target_mean_dist=0.1):
    # Start with predators in a tight triangle
    predator_positions = torch.Tensor([
        [0.0, 0.0],
        [0.05, 0.0],
        [0.025, 0.043]
    ]).to(pray_pos.device)

    # Place prey at center +/- small offset so mean distance ≈ target
    predator_positions = predator_positions.unsqueeze(0).repeat(pray_pos.shape[0], 1, 1) + pray_pos.unsqueeze(1).repeat(
        1, 3, 1)

    objective_pos = predator_positions
    objective_vel = torch.zeros_like(predator_positions)

    indices = [[2, 3, 4, 5], [0, 1, 4, 5], [0, 1, 2, 3]]
    # for i in range(3):
    #     s = i * 2
    #     exclude = [s, s + 1]
    #     indices.append([j for j in range(2 * 3) if j not in exclude])
    #
    relative_other_pos = predator_positions.reshape(predator_positions.shape[0], 1, -1).repeat(1, predator_positions.shape[1], 1) - objective_pos.repeat(1, 1, 3)

    indices = torch.tensor(indices, device=relative_other_pos.device)
    indices = indices.unsqueeze(0).expand(relative_other_pos.shape[0], -1, -1)

    relative_other_pos = torch.gather(relative_other_pos, 2, indices)

    relative_pray_pos = []
    for i in range(predator_positions.shape[0]):
        for j in range(predator_positions.shape[1]):
            relative_pray_pos.append(pray_pos[i] - objective_pos[i][j])

    relative_prey_pos = torch.stack(relative_pray_pos).reshape(predator_positions.shape[0], 3, 2)

    return objective_pos, objective_vel, relative_other_pos, relative_prey_pos


def generate_objective_node_features(agents_pos):
    objective_pos, objective_vel, relative_other_pos, relative_prey_pos = generate_objective_state_predators(
        agents_pos[:, 3, :])

    return objective_pos, objective_vel, relative_other_pos, relative_prey_pos


def get_state_from_obs(obs, agent_group):
    # from a single observation get the full env state
    single_obs = obs[:, 0]
    current_agents_pos = single_obs["agent_pos"]
    other_agents_absolute_pos = single_obs["other_pos"] + current_agents_pos.repeat(1, 3)
    agents_absolute_pos = torch.cat([current_agents_pos.unsqueeze(1), other_agents_absolute_pos.view(-1, 3, 2)], dim=1)
    absolute_entity_pos = single_obs["entity_pos"] + current_agents_pos.repeat(1, 2)

    indices = []
    for i in range(4):
        s = i * 2
        exclude = [s, s + 1]
        indices.append([j for j in range(2 * 4) if j not in exclude])

    indices = torch.tensor(indices, device=obs.device)
    indices = indices.unsqueeze(0).expand(obs.shape[0], -1, -1)

    obs = {
        "agent_pos": agents_absolute_pos,
        "agent_vel": torch.zeros_like(agents_absolute_pos),
        "other_pos": torch.zeros_like(obs["other_pos"]),
        "other_vel": torch.zeros_like(agents_absolute_pos),
        "entity_pos": absolute_entity_pos.unsqueeze(1).repeat(1, 4, 1) - agents_absolute_pos.repeat(1, 1, 2),
    }

    # generate obs for all 4 agents (3 adversaries + 1 prey)
    tmp_other_pos = agents_absolute_pos.reshape(-1, 8).unsqueeze(1).repeat(1, 4, 1) - agents_absolute_pos[:].repeat(1,
                                                                                                                    1,
                                                                                                                    4)
    obs["other_pos"] = torch.gather(tmp_other_pos, 2, indices)

    if agent_group == "agent":
        # if we are the pray, let's reorder the obs such as I am  the last agent
        obs["agent_pos"] = torch.cat([obs["agent_pos"][:, 1:, :], obs["agent_pos"][:, :1, :]], dim=1)
        obs["agent_vel"] = torch.cat([obs["agent_vel"][:, 1:, :], obs["agent_vel"][:, :1, :]], dim=1)
        obs["other_pos"] = torch.cat([obs["other_pos"][:, 1:, :], obs["other_pos"][:, :1, :]], dim=1)
        obs["other_pos"][:, 0:3, :] = torch.cat([obs["other_pos"][:, 0:3, 2:], obs["other_pos"][:, 0:3, :2]], dim=2)
        # obs["other_pos"][:, 0:3, :][:, :, [0, 1, 2, 3, 4, 5]] = obs["other_pos"][:, 0:3, :][:, :, [4, 5, 2, 3, 0, 1]]
        obs["other_vel"] = torch.cat([obs["other_vel"][:, 1:, :], obs["other_vel"][:, :1, :]], dim=1)
        obs["entity_pos"] = torch.cat([obs["entity_pos"][:, 1:, :], obs["entity_pos"][:, :1, :]], dim=1)

    return obs


class SimpleTagObjectiveSharing(Model):
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
        ) - self.n_agents * 2  # we remove the "landmark_pos" from the input features
        self.activation_class = activation_class
        self.threshold = threshold

        self.raw_feature_encoder = Encoder(16, 128).to(self.device)
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

        self.graph_encoder = TagContrastiveModel(self.device).to(device=self.device)
        self.graph_encoder.load_state_dict(
            torch.load(
                "/home/aamato/Documents/marl/objective-based-marl/Tasks/SimpleTag/contrastive_model/tag_dict_contrastive_model_full.pth"))
        self.graph_encoder.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:

            # merge groups obs
            obs = tensordict.get(self.agent_group)["observation"]

            state_obs = get_state_from_obs(obs, self.agent_group)

            agents_pos, agents_vel, other_pos, other_vel, entity_pos = extract_features_from_obs(
                state_obs)

            agents_vel = torch.zeros(agents_pos.shape).to(agents_pos.device)
            other_vel = torch.zeros(agents_pos.shape).to(other_pos.device)

            batch_size = agents_pos.shape[:-2][0]

            # create objective node features
            objective_pos, objective_vel, relative_other_pos, relative_prey_pos = generate_objective_node_features(
                agents_pos)

            obs_dict = {
                "agent_pos": agents_pos,
                "agent_vel": agents_vel,
                "other_pos": other_pos,
                "other_vel": other_vel,
                "entity_pos": entity_pos
            }
            with torch.no_grad():
                h_agent_graph_metric = self.graph_encoder(obs_dict, (objective_pos, objective_vel, relative_other_pos, relative_prey_pos))

            # create obs for agents in objective position and objective

            obj_obs_dict = {
                "agent_pos": torch.cat([objective_pos, agents_pos[:, 3, :].unsqueeze(1)], dim=1),
                "agent_vel": torch.zeros(batch_size, 4, 2).to(agents_pos.device),
                "other_pos": torch.cat([torch.cat([relative_other_pos, relative_prey_pos], dim=2),
                                        (objective_pos[:, :3, :].reshape(batch_size, -1) - agents_pos[:, 3, :].repeat(1,
                                                                                                                      3)).unsqueeze(
                                            1)], dim=1),
                "other_vel": other_vel,
                "entity_pos": entity_pos
            }

            with torch.no_grad():
                h_objective_graph_metric = self.graph_encoder(obj_obs_dict,  (objective_pos, objective_vel, relative_other_pos, relative_prey_pos))

            distance = torch.pairwise_distance(h_agent_graph_metric, h_objective_graph_metric,
                                               keepdim=True).unsqueeze(1).repeat(1, self.n_agents, 1)

            c_reward = torch.zeros_like(distance)  # Initialize reward tensor

            stability_threshold = 1  # Distance where stability reward applies

            # Reward before reaching the stable zone (Linear Decay)
            c_reward[distance < self.threshold] = self.threshold - distance[distance < self.threshold]

            # Smooth transition to stable reward (Linear instead of exponential)
            close_enough = distance < stability_threshold
            c_reward[close_enough] = 100

            # agents feature encoding
            agents_features = torch.cat([
                agents_pos,
                agents_vel,
                other_pos,
                other_vel,
                entity_pos,
            ], dim=2).view(batch_size, 4, -1)

            h_agents_features_enc = self.raw_feature_encoder.forward(
                agents_features.view(-1, 16)).view(-1, 128)

            # agents gnn
            h_only_agents_unrolled = h_agents_features_enc.view(-1, 128)
            agents_pos_unrolled = agents_pos.view(-1, 2)

            agents_graph = generate_graph(batch_size=batch_size,
                                          node_features=h_only_agents_unrolled,
                                          node_pos=agents_pos_unrolled,
                                          edge_attr=None,
                                          n_agents=4,
                                          use_radius=True,
                                          device=self.device)

            h_agents_graph = self.agents_agents_gnn.forward(agents_graph.x,
                                                            agents_graph.edge_index,
                                                            agents_graph.edge_attr).view(batch_size, 4, -1)

            context_features = torch.cat(
                [
                    h_agent_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
                    h_objective_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
                    distance
                ], dim=2)

            context_encoded = self.context_feature_encoder.forward(context_features.view(-1, 257)).view(batch_size,
                                                                                                        self.n_agents,
                                                                                                        -1)

            single_contex_encoded = context_encoded[:, 0, :].unsqueeze(1).repeat(1, 4, 1)
            agents_final_features = torch.cat(
                [
                    single_contex_encoded,
                    h_agents_graph
                ], dim=2)

            if self.agent_group == "adversary":
                # We only need the first agent's output
                agents_final_features = agents_final_features[:, :3, :]
            elif self.agent_group == "agent":
                # We only need the last agent's output
                agents_final_features = agents_final_features[:, 3, :]
                c_reward = -c_reward

            res = self.final_mlp(agents_final_features.view(batch_size, self.n_agents, -1))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], c_reward)

        return tensordict


@dataclass
class SimpleTagObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    threshold: float = 12.0

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleTagObjectiveSharing
