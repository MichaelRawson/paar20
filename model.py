import torch
from torch.nn import BatchNorm1d, Dropout, Linear, Module, ModuleList
from torch.nn.functional import relu
from torch_geometric import nn as geom
from torch_geometric.utils import to_undirected

LAYERS = 16
K = 8

class FullyConnectedLayer(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        x = relu(x)
        x = self.linear(x)
        return x

class ConvLayer(Module):
    def __init__(self):
        super().__init__()
        self.conv_in = geom.GCNConv(K, K)
        self.conv_out = geom.GCNConv(K, K, flow='target_to_source')

    def forward(self, x, edge_index):
        x = relu(x)
        in_x = self.conv_in(x, edge_index)
        out_x = self.conv_out(x, edge_index)
        x = torch.cat((in_x, out_x), dim=1)
        return x

class DenseBlock(Module):
    def __init__(self, layers):
        super().__init__()
        self.fc = ModuleList([
            FullyConnectedLayer(2 * K * (layer + 1), K)
            for layer in range(layers)
        ])
        self.conv = ModuleList([
            ConvLayer()
            for _ in range(layers)
        ])

    def forward(self, x, edge_index):
        outputs = [x]
        for conv, fc in zip(self.conv, self.fc):
            combined = torch.cat(outputs, dim=1)
            x = fc(combined)
            x = conv(x, edge_index)
            outputs.append(x)

        return torch.cat(outputs, dim=1)

class Actor(Module):
    def __init__(self, features):
        super().__init__()
        self.fc = FullyConnectedLayer(features, 128)
        self.policy = FullyConnectedLayer(128, 1)

    def forward(self, actions):
        return self.policy(self.fc(actions)).reshape(-1)

class Critic(Module):
    def __init__(self, features):
        super().__init__()
        self.fc = FullyConnectedLayer(2 * features, 128)
        self.value = FullyConnectedLayer(128, 1)

    def forward(self, clauses):
        max_pooled = clauses.max(dim=0)[0]
        mean_pooled = clauses.mean(dim=0)
        pooled = torch.cat([max_pooled, mean_pooled], dim=0)
        return self.value(self.fc(pooled)).reshape(())

class Model(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(11, 2 * K)
        self.dense = DenseBlock(LAYERS)
        self.actor = Actor(2 * K * (LAYERS + 1))
        self.critic = Critic(2 * K * (LAYERS + 1))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        action_index = data.action_index
        clause_index = data.clause_index

        x = self.input(x)
        x = self.dense(x, edge_index)
        actions = x[data.action_index]
        clauses = x[data.clause_index]
        return (self.actor(actions), self.critic(clauses))
