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


def extract_features_from_obs(obs):
    current_obs = obs  # [batch, 3]
    target = obs[..., -3:]  # the last 3
    return current_obs, target


def generate_objective_node_features(targets):
    obj_feature = torch.cat([targets, targets], dim=-1)
    return obj_feature


class LoadBalancingObjectiveSharing(Model):
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

        # self.context_feature_encoder = Encoder(257, 32).to(self.device)

        self.agents_agents_gnn = GATv2Conv(128, 128, 2, edge_dim=3).to(self.device)

        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=39,
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
        model_path = BASE_DIR / "Tasks/LoadBalancing/contrastive_model/lb_model_full_32.pth"

        self.contrastive_model = torch.load(model_path).to(
            device=self.device)
        self.contrastive_model.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            original_shape = tensordict.get(self.agent_group)["observation"].shape
            obs = tensordict.get(self.agent_group)["observation"]
            current_obs, target = extract_features_from_obs(tensordict.get(self.agent_group)["observation"])
            batch_size = current_obs.shape[0]

            if len(current_obs.shape) == 2:
                current_obs = current_obs.unsqueeze(0)
                target = target.unsqueeze(0)
                obs = obs.unsqueeze(0)
                batch_size = 1

            with torch.no_grad():
                final_emb, h_state_emb, h_object_emb = self.contrastive_model(obs)

            similarity = torch.nn.functional.cosine_similarity(h_state_emb,
                                                               h_object_emb,
                                                               dim=-1).unsqueeze(1).repeat(1, self.n_agents, 1)

            context_features = torch.cat([
                h_state_emb.repeat(1, self.n_agents, 1),
                h_object_emb.repeat(1, self.n_agents, 1),
                similarity], dim=2).view(batch_size, self.n_agents, -1)

            agents_final_features = torch.cat(
                [
                    obs,
                    context_features,
                ], dim=2)

            res = self.final_mlp(agents_final_features.view(batch_size, self.n_agents, -1))

        shape = []
        for element in original_shape:
            shape.append(element)
        shape = shape[:-1]
        shape.append(-1)

        tensordict.set(self.out_keys[0], res.reshape(tuple(shape)))
        tensordict.set(self.out_keys[1], similarity.reshape(tuple(shape)))

        return tensordict


@dataclass
class LoadBalancingObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return LoadBalancingObjectiveSharing
