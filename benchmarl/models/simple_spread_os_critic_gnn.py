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

from Tasks.SimpleSpread.contrastive_model.model.graph_contrastive_model import SimpleSpreadGraphContrastiveModel
from benchmarl.models.simple_spread_objective_sharing import split_spread_observation, create_objective_features
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
        self.linear2 = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        l1 = torch.relu(self.linear(x))
        l2 = torch.relu(self.linear1(l1))
        l3 = self.linear2(l2)

        return l3


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


class SimpleSpreadOSCriticGnn(Model):
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
        # self.in_keys.remove(('agents', 'observation', 'landmark_pos'))
        self.reduced_keys = self.in_keys.copy()
        self.reduced_keys.remove(('agents', 'observation', 'landmark_pos'))

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        ) - self.n_agents * 2  # we remove the "landmark_pos" from the input features

        self.output_features = self.output_leaf_spec.shape[-1]
        # torch.manual_seed(0)
        self.mlp = Encoder(51, self.output_features).to(self.device)

        self.mlp = MultiAgentMLP(
            n_agent_inputs=3,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents * 2,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            **kwargs,
        )


        # 4) Pre‑trained contrastive model providing a *global* context vector
        base_dir = Path(__file__).resolve().parents[3]
        model_path = (
                base_dir
                / "Tasks"
                / "SimpleSpread"
                / "contrastive_model"
                / "cosine_graph_model.pth"
        )
        self.contrastive_model = SimpleSpreadGraphContrastiveModel(self.n_agents, self.device)
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
            final_emb, final_emb_2, state_emb, obj_emb = self.contrastive_model(
                tensordict.get("agents")["observation"].view(-1, self.n_agents))

        similarity = torch.nn.functional.cosine_similarity(final_emb,
                                                           final_emb_2,
                                                           dim=-1)
        context = torch.cat([
            state_emb,
            obj_emb,
            similarity.unsqueeze(1)], dim=-1)
        # ---------------- 1. Parse observation ---------------- #
        obs = tensordict.get("agents")["observation"]

        (
            agents_pos,
            _agents_vel,  # Velocities unused for now
            landmark_pos,
            _rel_landmarks_pos,
            _rel_other_pos,
        ) = split_spread_observation(obs)

        batch_size = agents_pos.size(0)

        # ---------------- 2. Build *objective* state ------------ #
        (
            obj_pos,
            _obj_vel,
            _obj_rel_landmarks_pos,
            _obj_rel_other_pos,
        ) = create_objective_features(landmark_pos.view(-1, obs.shape[-1], 6), obs.shape[-1])

        # ---------------- 2. Encode graph nodes ----------------- #
        # type feature helps GNN distinguish agent vs. landmark

        shape = []
        for i in obs.shape:
            shape.append(i)
        shape.append(1)
        agent_type = torch.zeros(tuple(shape), device=self.device)
        obj_type = torch.ones_like(agent_type)

        shape[-1] = -1
        cur_nodes = torch.cat([agents_pos.view(shape), obj_pos.view(shape)], dim=-2)  # (B, 2N, 2)
        cur_types = torch.cat([agent_type, obj_type], dim=-2)  # (B, 2N, 1)
        shape[-2] = self.n_agents * 2
        cur_feats = torch.cat([cur_nodes, cur_types], dim=-1)  # (B*2N, 3)
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
class SimpleSpreadOSCriticGnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return SimpleSpreadOSCriticGnn
