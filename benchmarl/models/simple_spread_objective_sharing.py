from __future__ import annotations

from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Type, Sequence, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from tensordict import TensorDictBase
from torchrl.modules import MultiAgentMLP

from Tasks.SimpleSpread.contrastive_model.model.graph_contrastive_model import MLPEncoder, \
    SimpleSpreadGraphContrastiveModel
from Tasks.SimpleSpread.utils.utils import generate_graph, layer_init
from benchmarl.models.common import Model, ModelConfig

# --------------------------------------------------------------------------- #
#                               Helper Modules                                #
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    """One‑layer MLP used to embed raw node features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = layer_init(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        return F.relu(self.linear(x))


def split_spread_observation(obs: TensorDictBase):
    """Extract tensors of interest from a SimpleSpread observation dict.

    Returns:
        Tuple containing:
            agents_pos (B, N, 2)
            agents_vel (B, N, 2)
            landmarks_pos (B, N, 2)
            rel_landmarks_pos (B, N, 2)
            rel_other_pos (B, N, 2)
    """
    return (
        obs["agent_pos"],
        obs["agent_vel"],
        obs["landmark_pos"],
        obs["relative_landmark_pos"],
        obs["other_pos"],
    )


def create_objective_features(
    landmark_pos: torch.Tensor,
    n_agents: int,
):
    """Generate 'ideal' (objective) features where each agent sits on a landmark.

    Args:
        landmark_pos: (B, N, 2) absolute landmark positions.
        n_agents:     Number of agents (== number of landmarks in SimpleSpread).

    Returns:
        Tuple of tensors describing the objective state:
            objective_pos               (B, N, 2)
            objective_vel               (B, N, 2) – always zero
            rel_landmarks_pos_objective (B, N, 2)
            rel_other_pos_objective     (B, N, N‑1, 2)
    """
    bsz, n_landmarks, _ = landmark_pos.shape
    objective_pos = landmark_pos[:, 1, :].view(-1, n_landmarks, 2).clone()
    objective_vel = torch.zeros_like(objective_pos)

    # Position of every landmark relative to every other landmark
    rel_landmarks_pos = landmark_pos - objective_pos.repeat(1, 1, n_landmarks)

    # For each agent i remove its self‑distance to produce "other landmark" features
    indices = []
    for i in range(n_agents):
        start = i * 2
        indices.append(
            [j for j in range(2 * n_agents) if j not in (start, start + 1)]
        )

    idx = torch.tensor(indices, device=landmark_pos.device)  # (N, 2N‑2)
    idx = idx.unsqueeze(0).expand(bsz, -1, -1)               # (B, N, 2N‑2)
    rel_other_pos = torch.gather(rel_landmarks_pos, 2, idx)

    return objective_pos, objective_vel, rel_landmarks_pos, rel_other_pos


# --------------------------------------------------------------------------- #
#                            Main Agent‑Level Model                           #
# --------------------------------------------------------------------------- #

class SimpleSpreadObjectiveSharing(Model):
    """Actor network that augments each agent’s input with a learned
    representation of the *objective* (all agents sitting on landmarks)."""

    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int] = MISSING,
        layer_class: Type[nn.Module] = MISSING,
        **kwargs,
    ):
        # Initialise BenchMARL base Model
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

        self.activation_class = activation_class
        self.output_features = self.output_leaf_spec.shape[-1]
        # Remove landmark positions (2 dims per agent) from the raw observation feature count
        self.input_features = (
            sum(spec.shape[-1] for spec in self.input_spec.values(True, True))
            - self.n_agents * 2
        )

        # ----------------------------- Sub‑modules -------------------------- #
        # 1) Graph‑level communication between agents
        self.gnn = GATv2Conv(16, 16, heads=1, edge_dim=3).to(self.device)

        # 2) Final per‑agent policy head
        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=81,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=activation_class,
            layer_class=layer_class,
            num_cells=num_cells,
        )

        # 3) Node encoder shared by agents & landmarks – (x, y, type) → 16‑D
        self.node_encoder = MLPEncoder(input_size=3, output_size=16).to(self.device)

        # 4) Pre‑trained contrastive model providing a *global* context vector
        base_dir = Path(__file__).resolve().parents[3]
        model_path = (
            base_dir
            / "Tasks"
            / "SimpleSpread"
            / "contrastive_model"
            / "model_epoch_28.pth"
        )
        self.contrastive_model = SimpleSpreadGraphContrastiveModel(self.n_agents, self.device)
        self.contrastive_model.load_state_dict(torch.load(model_path))
        self.contrastive_model.eval()

    # ----------------------------- Forward Pass ------------------------------ #

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.input_has_agent_dim:
            raise ValueError("Model expects per‑agent dimension in the input.")

        # ---------------- 1. Parse observation ---------------- #
        obs = tensordict.get("agents")["observation"]
        (
            agents_pos,
            _agents_vel,   # Velocities unused for now
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
        ) = create_objective_features(landmark_pos, self.n_agents)

        # ---------------- 3. Encode graph nodes ----------------- #
        # type feature helps GNN distinguish agent vs. landmark
        agent_type = torch.zeros((batch_size, self.n_agents, 1), device=self.device)
        lm_type = torch.ones_like(agent_type)

        cur_nodes = torch.cat([agents_pos, obj_pos], dim=1)        # (B, 2N, 2)
        cur_types = torch.cat([agent_type, lm_type], dim=1)
        cur_feats = torch.cat([cur_nodes, cur_types], dim=-1).view(-1, 3)  # (B*2N, 3)

        node_embeddings = self.node_encoder(cur_feats)             # (B*2N, 16)

        # ---------------- 4. Global contrastive context ---------- #
        with torch.no_grad():
            final_emb, final_emb_2, *_ = self.contrastive_model(obs)

        similarity = F.cosine_similarity(final_emb, final_emb_2, dim=-1).unsqueeze(1).unsqueeze(1)
        similarity = similarity.repeat(1, self.n_agents, 1)         # (B, N, 1)

        # ---------------- 5. Build batched graph ---------------- #
        num_total_nodes = self.n_agents * 2
        graph_repr = generate_graph(
            batch_size=batch_size,
            node_features=node_embeddings,
            node_pos=cur_nodes.view(-1, 2),
            edge_attr=None,
            n_agents=num_total_nodes,
            device=self.device,
        )

        cur_h = self.gnn(graph_repr.x, graph_repr.edge_index, graph_repr.edge_attr).view(batch_size, num_total_nodes, -1)

        # ---------------- 6. Concatenate all features ------------ #
        context = torch.cat(
            [
                final_emb.unsqueeze(1).repeat(1, self.n_agents, 1),
                final_emb_2.unsqueeze(1).repeat(1, self.n_agents, 1),
                similarity,
            ],
            dim=2,
        )                                                           # (B, N, 65)

        agent_inputs = torch.cat([cur_h[:, :self.n_agents, :], context], dim=2)      # (B, N, 81)

        # ---------------- 7. Per‑agent policy head --------------- #
        actions = self.final_mlp(agent_inputs.view(batch_size, self.n_agents, -1))

        # ---------------- 8. Write outputs into TensorDict ------- #
        tensordict.set(self.out_keys[0], actions)
        tensordict.set(self.out_keys[1], similarity)

        return tensordict


# --------------------------------------------------------------------------- #
#                           Hydra / YAML Config Glue                          #
# --------------------------------------------------------------------------- #

@dataclass
class SimpleSpreadObjectiveSharingConfig(ModelConfig):
    """Hydra config schema for :class:`SimpleSpreadObjectiveSharing`."""

    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_kwargs: Optional[dict] = None
    norm_class: Optional[Type[nn.Module]] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return SimpleSpreadObjectiveSharing
