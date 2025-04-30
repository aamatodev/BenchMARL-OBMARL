from __future__ import annotations

from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Type, Sequence, Optional

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv

from cmodels.scl_model import SCLModel
from cmodels.scl_model_v5 import SCLModelV5
from torchrl.data import Composite, Unbounded

from benchmarl.models import Gnn, GnnConfig, DeepsetsConfig, Deepsets
from benchmarl.models.common import Model, ModelConfig

from torchrl.data import Composite, Unbounded, ReplayBuffer, LazyTensorStorage

from cmodels.scl_model_v2 import SCLModelv2
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
    landmarks_pos = obs["landmark_pos"]
    relative_landmarks_pos = obs["relative_landmark_pos"]
    relative_other_pos = obs["other_pos"]

    return agents_pos, agents_vel, landmarks_pos, relative_landmarks_pos, relative_other_pos


def generate_objective_node_features(landmark_pos, n_agents=3):
    # in simple spread, the objective is reached once all the landmarks are covered. This happens when the agents
    # positions are equals to the landmarks positions
    n_landmark = landmark_pos.shape[1]
    objective_pos = landmark_pos[:, 1, :].view(-1, n_landmark, 2).clone().detach()
    objective_vel = torch.zeros_like(objective_pos)

    relative_landmarks_pos = landmark_pos - objective_pos.repeat(1, 1, n_landmark)

    # Dynamically generate indices for n agents
    indices = []
    for i in range(n_agents):
        s = i * 2
        exclude = [s, s + 1]
        indices.append([j for j in range(2 * n_agents) if j not in exclude])

    indices = torch.tensor(indices, device=landmark_pos.device)
    indices = indices.unsqueeze(0).expand(landmark_pos.shape[0], -1, -1)
    # Use `gather` to apply the indexing along the last dimension
    relative_other_pos = torch.gather(relative_landmarks_pos, 2, indices)

    return objective_pos, objective_vel, relative_landmarks_pos, relative_other_pos


class SimpleSpreadObjectiveSharing(Model):
    def __init__(
            self,
            activation_class: Type[nn.Module],
            threshold: float,
            stability: float,
            num_cells: Sequence[int] = MISSING,
            layer_class: Type[nn.Module] = MISSING,
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
        self.stability = stability

        # self.context_feature_encoder = Encoder(257, 32).to(self.device)

        self.agents_agents_gnn = GATv2Conv(128, 128, 2, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=14,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=activation_class,
            layer_class=layer_class,
            num_cells=num_cells,
        )

        # self.graph_encoder = SCLModelV5(self.device).to(device=self.device)
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        model_path = BASE_DIR / "contrastive_learning/siamese_model_final.pth"

        self.graph_encoder = torch.load(model_path).to(device=self.device)
        self.graph_encoder.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            agents_pos, agents_vel, landmark_pos, relative_landmarks_pos, relative_other_pos = extract_features_from_obs(
                tensordict.get("agents")["observation"])
            batch_size = agents_pos.shape[:-2][0]

            objective_pos, objective_vel, objective_relative_landmarks_pos, objective_relative_other_pos = generate_objective_node_features(
                landmark_pos, self.n_agents)

            with torch.no_grad():
                h_agent_graph_metric = self.graph_encoder(tensordict.get("agents")["observation"])

            # # create obs for agents in objective position and objective
            # obs = dict()
            # obs["agent_pos"] = objective_pos.view(-1, self.n_agents, 2)
            # obs["landmark_pos"] = landmark_pos
            # obs["agent_vel"] = objective_vel.view(batch_size, self.n_agents, 2)
            # obs["relative_landmark_pos"] = objective_relative_landmarks_pos
            # obs["other_pos"] = objective_relative_other_pos
            #
            # with torch.no_grad():
            #     h_objective_graph_metric = self.graph_encoder(obs)
            #
            # distance = torch.pairwise_distance(h_agent_graph_metric, h_objective_graph_metric,
            #                                    keepdim=True).unsqueeze(1).repeat(1, self.n_agents, 1)
            #
            # c_reward = torch.zeros_like(distance)  # Initialize reward tensor
            #
            # stability_threshold = self.stability  # Distance where stability reward applies
            #
            # # Reward before reaching the stable zone (Linear Decay)
            # c_reward[distance < self.threshold] = self.threshold - distance[distance < self.threshold]
            #
            # # Smooth transition to stable reward (Linear instead of exponential)
            # close_enough = distance < stability_threshold
            # c_reward[close_enough] = 100

            # agents feature encoding
            agents_features = torch.cat([
                agents_pos,
                agents_vel,
                relative_landmarks_pos,
                relative_other_pos], dim=2).view(batch_size, self.n_agents, -1)

            # context_features = torch.cat(
            #     [
            #         h_agent_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
            #         h_objective_graph_metric.unsqueeze(1).repeat(1, self.n_agents, 1),
            #         distance,
            #     ], dim=2)

            # context_encoded = self.context_feature_encoder.forward(context_features).view(batch_size,
            #                                                                               self.n_agents,
            #                                                                               -1)

            agents_final_features = torch.cat(
                [
                    agents_features,
                    # context_features,
                ], dim=2)

            res = self.final_mlp(agents_final_features.view(batch_size, self.n_agents, -1))

        tensordict.set(self.out_keys[0], res)
        tensordict.set(self.out_keys[1], 1 - h_agent_graph_metric.view(batch_size, 1, 1).repeat(1, 3, 1))

        return tensordict


@dataclass
class SimpleSpreadObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_kwargs: Optional[dict] = None
    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    threshold: float = 12.0
    stability: float = 0.2

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleSpreadObjectiveSharing
