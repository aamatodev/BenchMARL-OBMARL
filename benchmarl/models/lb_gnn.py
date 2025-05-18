from __future__ import annotations

from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Type, Sequence
import torch
from torch_geometric.nn import GATv2Conv

from Tasks.LoadBalancing.contrastive_model.model.lb_graph_contrastive_model import LbContrastiveGraphModel, MLPEncoder, \
    generate_graph
from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.model import layer_init
from tensordict import TensorDictBase
from torch import nn
import torch.nn.functional as F


class FinalEncoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 256)
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        l1 = torch.relu(self.linear(x))
        l2 = torch.relu(self.linear1(l1))
        l3 = torch.relu(self.linear2(l2))
        l4 = self.linear3(l3)

        return l4


class Encoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = layer_init(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        return self.linear(x)


def extract_features_from_obs(obs):
    current_obs = obs  # [batch, 3]
    target = obs[..., -3:]  # the last 3
    return current_obs, target


def generate_objective_node_features(targets):
    obj_feature = torch.cat([targets, targets], dim=-1)
    return obj_feature


class LoadBalancingGnn(Model):
    def __init__(
            self,
            activation_class: Type[nn.Module],
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

        # ----------------------------- Sub‑modules -------------------------- #

        # 1) Graph‑level communication between agents
        self.gnn = GATv2Conv(16, 16, heads=3, edge_dim=1).to(self.device)
        self.final_mlp = FinalEncoder(48, self.output_features).to(self.device)
        # 3) Node encoder shared by agents & landmarks – (x, y, type) → 16‑D
        self.node_encoder = MLPEncoder(input_size=2, output_size=16).to(self.device)

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict["agent"]["observation"]
        current_status, target_status = extract_features_from_obs(obs)

        if len(obs.shape) == 2:
            # If the observation is 2D, we need to add a dimension
            obs = obs.unsqueeze(0)
            current_status = current_status.unsqueeze(0)

        batch_size = current_status.shape[0]
        agent_type = torch.zeros((batch_size, self.n_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        current_status = current_status[..., 0, :]
        cur_nodes_f = current_status.view(batch_size, self.n_agents * 2, 1)

        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes_f, cur_types], dim=-1).view(-1, 2)

        # Encode node features
        cur_encoded = self.node_encoder(cur_feats)

        num_total_nodes = self.n_agents * 2  # agents + landmarks per batch

        cur_graph = generate_graph(
            batch_size=batch_size,
            node_features=cur_encoded,
            node_pos=cur_nodes_f.reshape(-1, 1),
            edge_attr=None,
            n_agents=num_total_nodes,
            device=self.device,
        )

        # Shared GAT‑v2 over both graphs
        cur_h = self.gnn(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)

        agent_inputs = cur_h.view((obs.shape[0], self.n_agents * 2, -1))[:, :self.n_agents, :]  # (B, N, 81)

        # ---------------- 7. Per‑agent policy head --------------- #
        actions = self.final_mlp(agent_inputs.view(batch_size, self.n_agents, -1))

        # ---------------- 8. Write outputs into TensorDict ------- #
        if len(tensordict["agent"]["observation"].shape) == 2:
            # If the observation is 2D, we need to remove the dimension
            actions = actions.squeeze(0)

        tensordict.set(self.out_keys[0], actions)

        return tensordict


@dataclass
class LoadBalancingGnnConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return LoadBalancingGnn
