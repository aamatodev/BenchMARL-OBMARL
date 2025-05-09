from __future__ import annotations

from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv

from Tasks.SimpleTag.contrastive_model.model.tag_contrastive_model import TagContrastiveModel
from benchmarl.models import GnnConfig, MlpConfig, Mlp, Gnn
from benchmarl.models.common import Model, ModelConfig
from torchrl.data import Composite, Unbounded

from tensordict import TensorDictBase, TensorDict
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
    relative_other_pos = predator_positions.reshape(predator_positions.shape[0], 1, -1).repeat(1,
                                                                                               predator_positions.shape[
                                                                                                   1],
                                                                                               1) - objective_pos.repeat(
        1, 1, 3)

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


import torch

# constant “other‑agent” gather indices (shape = 4 agents × 6 coords)
_GATHER_IDX = torch.tensor(
    [[2, 3, 4, 5, 6, 7],  # agent‑0 wants coords of agents‑1 & 2 & 3
     [0, 1, 4, 5, 6, 7],  # agent‑1 …
     [0, 1, 2, 3, 6, 7],  # agent‑2 …
     [0, 1, 2, 3, 4, 5]],  # agent‑3 …
    dtype=torch.long
)


def get_state_from_obs_v2(obs: dict, agent_group: str):
    """
    Convert a batch of PettingZoo‑style observations into a full environment state
    for *all four* agents (3 predators + 1 prey).

    Args
    ----
    obs         : dict with keys "agent_pos", "other_pos", "entity_pos", …
                  Shapes follow the user’s original code.
    agent_group : "adversary" | "agent"  (prey == "agent")

    Returns
    -------
    A dict with exactly the same keys as before, but built via pure
    tensor operations (no Python loops).
    """
    # ── unpack the leading‑agent slice ───────────────────────────────────
    single = obs[:, 0]  # (B, …)
    p0_pos = single["agent_pos"]  # (B, 2)
    # these arrive flattened: (B, 6) and (B, 4)
    other_r = single["other_pos"].reshape(-1, 3, 2)  # -> (B, 3, 2)
    entity_r = single["entity_pos"].reshape(-1, 2, 2)  # -> (B, 2, 2)

    B, _ = p0_pos.shape
    device = p0_pos.device

    # ── absolute positions for *all* 4 agents ───────────────────────────
    agents_abs = torch.cat([p0_pos.unsqueeze(1),  # (B, 1, 2)
                            other_r + p0_pos.unsqueeze(1)],  # (B, 3, 2)
                           dim=1)  # (B, 4, 2)

    # ── absolute positions for the 2 entities ───────────────────────────
    entity_abs = entity_r + p0_pos.unsqueeze(1)  # (B, 2, 2)

    # ── “other‑agent” positions relative to each agent ──────────────────
    flat = agents_abs.view(B, 8).unsqueeze(1).expand(-1, 4, 8)
    rel_all = flat - agents_abs.repeat(1, 1, 4)  # (B, 4, 8)
    gather_idx = _GATHER_IDX.to(device).expand(B, -1, -1)
    other_rel = torch.gather(rel_all, 2, gather_idx)  # (B, 4, 6)

    # ── entity positions relative to each agent ─────────────────────────
    entity_rel = entity_abs.view(B, 4).unsqueeze(1) - agents_abs.repeat(1, 1, 2)  # (B, 4, 4)

    zeros_vel = agents_abs.new_zeros(B, 4, 2)  # stationary

    state = {
        "agent_pos": agents_abs,
        "agent_vel": zeros_vel.clone(),
        "other_pos": other_rel,
        "other_vel": zeros_vel,
        "entity_pos": entity_rel,
    }

    # ── if we’re the prey, rotate so prey is last & fix “other_pos” order ─
    if agent_group == "agent":
        for k in ("agent_pos", "agent_vel", "other_pos", "other_vel", "entity_pos"):
            state[k] = torch.roll(state[k], shifts=-1, dims=1)

        # within each (6‑long) other_pos vector, rotate pairs to match new ordering
        state["other_pos"][:, :3] = torch.roll(state["other_pos"][:, :3], shifts=-2, dims=2)

    return state


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

    indices = torch.tensor(indices, device=current_agents_pos.device)
    indices = indices.unsqueeze(0).expand(obs.shape[0], -1, -1)

    # if agent_group == "adversary":
    #     other_vel = obs["other_vel"][:, :1, :]
    #     agent_vel = torch.cat([obs["agent_vel"], other_vel], dim=1)
    # else:
    #     other_vel = obs["other_vel"][:, 3:, :]

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
        self.activation_class = activation_class
        self.num_cells = kwargs.pop("num_cells", [256, 256])

        self.raw_feature_encoder = Encoder(16, 128).to(self.device)
        self.context_feature_encoder = Encoder(257, 32).to(self.device)

        # self.gnn_input_spec = Composite(
        #     {
        #         self.agent_group: Composite(
        #             {"input": Unbounded(shape=(self.n_agents, 128)), },
        #             shape=(self.n_agents,),
        #         )
        #     }
        # )
        #
        # self.gnn_output_spec = Composite(
        #     {
        #         self.agent_group: Composite(
        #             {"output": Unbounded(shape=(self.n_agents, 128)), },
        #             shape=(self.n_agents,),
        #         )
        #     }
        # )

        # self.agents_agents_gnn = Gnn(topology="from_pos",
        #                              edge_radius=0.5,
        #                              self_loops=True,
        #                              gnn_class=torch_geometric.nn.conv.GATv2Conv,
        #                              gnn_kwargs={"heads": 1},
        #                              position_key="agent_pos",
        #                              exclude_pos_from_node_features=True,
        #                              velocity_key=None,
        #                              pos_features=2,
        #                              vel_features=0,
        #                              agent_group=self.agent_group,
        #                              input_has_agent_dim=self.input_has_agent_dim,
        #                              n_agents=self.n_agents,
        #                              centralised=self.centralised,
        #                              share_params=self.share_params,
        #                              device=self.device,
        #                              is_critic=self.is_critic,
        #                              input_spec=self.gnn_input_spec,
        #                              output_spec=self.gnn_output_spec,
        #                              action_spec=self.action_spec,
        #                              model_index=self.model_index,
        #                              )

        self.final_mlp_input_spec = Composite(
            {
                self.agent_group: Composite(
                    {"input": Unbounded(shape=(self.n_agents, 49)), },
                    shape=(self.n_agents,),
                )
            }
        )

        self.final_mlp = Mlp(
            input_spec=self.final_mlp_input_spec,
            output_spec=self.output_spec,
            agent_group=self.agent_group,
            input_has_agent_dim=self.input_has_agent_dim,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            action_spec=self.action_spec,
            model_index=self.model_index,
            is_critic=self.is_critic,
            activation_class=activation_class,
            num_cells=self.num_cells,
        )

        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        model_path = BASE_DIR / "Tasks/SimpleTag/contrastive_model/tag_cosine_model.pth"
        self.graph_encoder = torch.load(model_path).to(device=self.device)
        self.graph_encoder.eval()

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Input has multi-agent input dimension
        if self.input_has_agent_dim:
            # merge groups obs
            obs = tensordict.get(self.agent_group)["observation"]
            batch_size = obs.shape[0]
            state_obs = get_state_from_obs_v2(obs, self.agent_group)

            with torch.no_grad():
                final_emb, state_emb, obj_emb = self.graph_encoder(state_obs)

            similarity = torch.nn.functional.cosine_similarity(state_emb,
                                                               obj_emb,
                                                               dim=-1).unsqueeze(1).repeat(1, self.n_agents, 1)

            # agents feature encoding
            agents_features = torch.cat([
                obs["agent_pos"],
                obs["agent_vel"],
                obs["other_pos"],
                obs["other_vel"],
                obs["entity_pos"],
            ], dim=2).view(batch_size, self.n_agents, -1)

            context_features = torch.cat(
                [
                    state_emb.repeat(1, self.n_agents, 1),
                    obj_emb.repeat(1, self.n_agents, 1),
                    similarity,
                ], dim=2)

            agents_final_features = torch.cat(
                [
                    agents_features,
                    context_features,
                ], dim=2)

            mlp_input = TensorDict({self.agent_group: {
                "input": agents_final_features}}, )

            res = self.final_mlp(mlp_input)

        tensordict.set(self.out_keys[0], res.get((self.agent_group, "logits")))
        tensordict.set(self.out_keys[1], similarity if self.agent_group == "adversary" else -similarity)

        return tensordict


@dataclass
class SimpleTagObjectiveSharingConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING
    num_cells: list[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return SimpleTagObjectiveSharing
