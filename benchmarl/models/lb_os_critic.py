#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Optional, Sequence, Type

import torch

from Tasks.LoadBalancing.contrastive_model.model.lb_graph_contrastive_model import LbContrastiveGraphModel
from benchmarl.models.simple_tag_objective_sharing import get_state_from_obs_v2
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig


class Encoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 256)
        self.linear1 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        l1 = torch.relu(self.linear(x))
        l2 = torch.relu(self.linear1(l1))
        l3 = self.linear3(l2)

        return l3


def extract_features_from_obs(obs):
    current_obs = obs  # [batch, 3]
    target = obs[..., -3:]  # the last 3
    return current_obs, target


def generate_objective_node_features(targets):
    obj_feature = torch.cat([targets, targets], dim=-1)
    return obj_feature


class LoadBalancingMlp(Model):
    """Multi layer perceptron model.

    Args:
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
        layer_class (Type[nn.Module]): class to be used for the linear layers;
        activation_class (Type[nn.Module]): activation class to be used.
        activation_kwargs (dict, optional): kwargs to be used with the activation class;
        norm_class (Type, optional): normalization class, if any.
        norm_kwargs (dict, optional): kwargs to be used with the normalization layers;

    """

    def __init__(
            self,
            **kwargs,
    ):
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

        self.input_features = 39
        self.output_features = self.output_leaf_spec.shape[-1]

        self.mlp = Encoder(77, self.output_features).to(self.device)

        # 4) Pre‑trained contrastive model providing a *global* context vector
        base_dir = Path(__file__).resolve().parents[3]
        model_path = (
                base_dir
                / "Tasks"
                / "LoadBalancing"
                / "contrastive_model"
                / "cosine_graph_model.pth"
        )
        self.contrastive_model = LbContrastiveGraphModel(self.n_agents, self.device)
        self.contrastive_model.load_state_dict(torch.load(model_path))
        self.contrastive_model.eval()

    def _perform_checks(self):
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                    not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"MLP inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"MLP input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the MLP input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
                self.output_has_agent_dim
                and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        # ---------------- 1. Global contrastive context ---------- #
        with torch.no_grad():
            final_emb, state_emb, obj_emb = self.contrastive_model(
                tensordict.get("agent")["observation"].view(-1, self.n_agents,
                                                            tensordict.get("agent")["observation"].shape[-1]))

        similarity = torch.nn.functional.cosine_similarity(state_emb,
                                                           obj_emb,
                                                           dim=-1)
        context = torch.cat([
            state_emb,
            obj_emb,
            similarity.unsqueeze(1)], dim=-1)

        # ---------------- 2 Parse observation ---------------- #

        obs = tensordict["agent"]["observation"]
        current_status, target_status = extract_features_from_obs(obs)
        batch_size = current_status.shape[0]

        current_status = current_status[..., 0, :]
        cur_nodes_f = current_status.reshape(-1, self.n_agents * 2, 1)

        agent_type = torch.zeros((cur_nodes_f.shape[0], self.n_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes_f, cur_types], dim=-1)

        shape = []
        for i in obs.shape:
            shape.append(i)

        shape[-1] = -1
        shape[-2] = 1
        cur_feats = cur_feats.view(tuple(shape))
        cur_feats = torch.cat([cur_feats, context.view(tuple(shape))], dim=-1)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            res = self.mlp.forward(cur_feats)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(input) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class LoadBalancingMlpConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return LoadBalancingMlp
