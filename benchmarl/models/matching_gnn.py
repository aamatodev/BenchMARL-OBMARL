from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from torchrl.data import Composite, Unbounded

from benchmarl.models import Gnn, GnnConfig, DeepsetsConfig, Deepsets
from benchmarl.models.common import Model, ModelConfig
from tensordict import TensorDictBase, TensorDict
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    """Encoder for initial state of DGN"""

    def __init__(self, num_input_feature, num_output_feature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = layer_init(nn.Linear(num_input_feature, num_output_feature))

    def forward(self, obs):
        embedding = F.relu(self.l1(obs))
        return embedding


class MatchingGNN(Model):
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

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )

        # And access some of the ones already available to your module
        _ = self.input_spec  # Like its input_spec
        _ = self.output_spec  # or output_spec
        _ = self.action_spec  # the action spec of the env
        _ = self.agent_group  # the name of the agent group the model is for
        _ = self.n_agents  # or the number of agents this module is for
        _ = self.input_has_agent_dim
        _ = self.share_params
        _ = self.centralised
        _ = self.output_has_agent_dim

        self.output_features = self.output_leaf_spec.shape[-1]

        self.activation_function = activation_class

        # Define CompositeSpecs for the landmark gnn
        landmark_position_spec = Composite(
            {
                "nodes": Composite(
                    {"node_pos": Unbounded(shape=(self.n_agents, 2))},
                    shape=(self.n_agents,),
                )
            }
        )

        # Define CompositeSpecs for the agents gnn
        agents_input_spec = Composite(
            {
                "agents": Composite(
                    {"agent_pos": Unbounded(shape=(self.n_agents, 2)),
                     "agent_vel": Unbounded(shape=(self.n_agents, 2)),
                     "relative_landmark_pos": Unbounded(shape=(self.n_agents, self.n_agents * 3)),
                     "similarity": Unbounded(shape=(self.n_agents, 1))},
                    shape=(self.n_agents,),
                )
            }
        )

        positional_output_spec = Composite(
            {
                "graph_latent_rep": Unbounded(shape=(self.n_agents, 8))
            }
        )

        agent_gnn_output_spec = Composite(
            {
                "graph_latent_rep": Unbounded(shape=(self.n_agents, 32))
            }
        )

        matching_input_spec = self.n_agents * 8

        self.node_pos_encoder = Encoder(2, 8).to(self.device)
        self.edge_encoder = Encoder(2, 8).to(self.device)
        self.matching_gnn = GATv2Conv(3, 128, 1, edge_dim=3).to(self.device)
        self.agent_gnn = GATv2Conv(128, 128, 1, edge_dim=3).to(self.device)
        self.Final_encoder = Encoder(128, 9).to(self.device)

        # if self.input_has_agent_dim and not self.centralised:
        # self.matching_gnn = Gnn_v2(
        #     topology="from_pos",
        #     self_loops=False,
        #     gnn_class=torch_geometric.nn.conv.GATv2Conv,
        #     gnn_kwargs={"aggr": "add"},
        #     position_key="node_pos",
        #     pos_features=2,
        #     velocity_key=None,
        #     vel_features=0,
        #     exclude_pos_from_node_features=False,
        #     edge_radius=0.5,
        #
        #     input_spec=landmark_position_spec,
        #     output_spec=positional_output_spec,
        #     agent_group="nodes",
        #     input_has_agent_dim=self.input_has_agent_dim,
        #     n_agents=self.n_agents,
        #     centralised=self.centralised,
        #     share_params=True,
        #     device=self.device,
        #     action_spec=self.action_spec,
        #     model_index=self.model_index,
        #     is_critic=self.is_critic,
        # )
        #
        # self.gnn_agents = Gnn(
        #     topology="from_pos",
        #     self_loops=False,
        #     gnn_class=torch_geometric.nn.conv.GATv2Conv,
        #     gnn_kwargs={"aggr": "add"},
        #     position_key="agent_pos",
        #     pos_features=2,
        #     velocity_key="agent_vel",
        #     vel_features=2,
        #     exclude_pos_from_node_features=False,
        #     edge_radius=0.5,
        #
        #     input_spec=agents_input_spec,
        #     output_spec=agent_gnn_output_spec,
        #     agent_group="agents",
        #     input_has_agent_dim=self.input_has_agent_dim,
        #     n_agents=self.n_agents,
        #     centralised=self.centralised,
        #     share_params=self.share_params,
        #     device=self.device,
        #     action_spec=self.action_spec,
        #     model_index=self.model_index,
        #     is_critic=self.is_critic,
        # )
        #
        # self.graph_embedding_mlp = MLP(in_features=matching_input_spec,
        #                                out_features=32,
        #                                depth=2,
        #                                device=self.device)
        #
        self.final_mlp = MultiAgentMLP(
            n_agent_inputs=128,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_function
        )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        matching_input_dict = Composite(
            {
                "nodes": Composite(
                    {"node_pos": Unbounded(shape=(self.n_agents, 2))},
                    shape=[],
                )
            }
        )

        agents_input_dict = Composite(
            {
                "agents": Composite(
                    {
                        "agent_pos": Unbounded(shape=(self.n_agents, 2)),
                        "agent_vel": Unbounded(shape=(self.n_agents, 2)),
                        "relative_landmark_pos": Unbounded(shape=(self.n_agents, self.n_agents * 3)),
                        "similarity": Unbounded(shape=(self.n_agents, 1))},
                    shape=[],
                )
            }
        )

        # Input has multi-agent input dimension
        if self.input_has_agent_dim:

            # create the agnet - objective graph
            batch_size = tensordict.get("agents")["observation"]["agent_pos"].shape[:-2][0]

            landmark_positions = tensordict.get("agents")["observation"]["landmark_pos"]
            landmark_positions = landmark_positions.view(-1, 2)

            landmark_eaten = tensordict.get("agents")["observation"]["landmark_eaten"]
            landmark_eaten = landmark_eaten.view(-1, 1)

            node_features = torch.cat([landmark_positions, landmark_eaten], dim=1)

            landmark_positions_ecoding = self.node_pos_encoder(landmark_positions).view(-1, 8)
            b = torch.arange(batch_size * self.n_agents, device=self.device)
            graphs = torch_geometric.data.Batch()
            graphs.ptr = torch.arange(0, (batch_size * self.n_agents + 1) * (self.n_agents+1), self.n_agents+1)
            graphs.batch = torch.repeat_interleave(b, self.n_agents + 1)

            graphs.x = node_features
            graphs.pos = landmark_positions
            graphs.edge_attr = None
            graphs.edge_index = torch_geometric.nn.pool.radius_graph(
                graphs.pos, batch=graphs.batch, r=1, loop=True
            )
            graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
            graphs = torch_geometric.transforms.Distance(norm=False)(graphs)

            h1 = F.relu(self.matching_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))
            # get the agents observation
            agent_objective_embedding_post_gnn = h1[0::5]

            # create the agent - agent graph
            agent_positions = tensordict.get("agents")["observation"]["agent_pos"]
            # matching_input_dict.get("nodes").set("node_pos", agent_positions)
            # agent_pos_graph_rep = self.matching_gnn.forward(matching_input_dict).get("graph_latent_rep").reshape(
            #     batch_size, -1).unsqueeze(1).expand(-1, 4, -1)
            b = torch.arange(batch_size, device=self.device)
            graphs = torch_geometric.data.Batch()
            graphs.ptr = torch.arange(0, (batch_size + 1) * self.n_agents, self.n_agents)
            graphs.batch = torch.repeat_interleave(b, self.n_agents)
            graphs.x = agent_objective_embedding_post_gnn
            graphs.pos = agent_positions.view(-1, 2)
            graphs.edge_attr = None
            graphs.edge_index = torch_geometric.nn.pool.radius_graph(
                graphs.pos, batch=graphs.batch, r=1, loop=True
            )
            graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
            graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
            h2 = F.relu(self.agent_gnn(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr))

            # objective_graph_embedding = self.graph_embedding_mlp.forward(objective_pos_graph_rep)
            # agent_graph_embedding = self.graph_embedding_mlp.forward(agent_pos_graph_rep)
            #
            # cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            #
            # similarity = cos(objective_graph_embedding, agent_graph_embedding).reshape(batch_size, -1, 1)
            #
            # agents_input_dict.get("agents").set("agent_pos", tensordict.get("agents")["observation"]["agent_pos"])
            # agents_input_dict.get("agents").set("agent_vel", tensordict.get("agents")["observation"]["agent_vel"])
            # agents_input_dict.get("agents").set("relative_landmark_pos",
            #                                     tensordict.get("agents")["observation"]["relative_landmark_pos"])
            # agents_input_dict.get("agents").set("similarity", similarity)
            #
            # res = self.gnn_agents.forward(agents_input_dict).get("graph_latent_rep")

            # for key, value in res.items():
            #     if key == ("agents", "action_value"):
            #         res = value

            # final_layer_input = torch.cat([objective_graph_embedding, agent_graph_embedding, similarity, res], dim=2)
            res = self.final_mlp.forward(h2.view(batch_size, self.n_agents, -1))

        # Input does not have multi-agent input dimension
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
class MatchingGNNConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml
    activation_class: Type[nn.Module] = MISSING

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return MatchingGNN
