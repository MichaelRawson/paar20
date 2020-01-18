import torch
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu
from torch_geometric import nn as geom
from torch_geometric.utils import degree

LAYERS = 32
K = 8

class GCNConv(geom.MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.weight = Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        x = self.weight(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

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
        self.conv_in = GCNConv(
            K, K,
        )
        self.conv_out = GCNConv(
            K, K,
            flow='target_to_source'
        )

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

class Q(Module):
    def __init__(self, features):
        super().__init__()
        self.fc = FullyConnectedLayer(features, 128)
        self.policy = FullyConnectedLayer(128, 1)

    def forward(self, actions):
        return self.policy(self.fc(actions)).reshape(-1)

class Model(Module):
    def __init__(self):
        super().__init__()
        self.input = Linear(11, 2 * K, bias=False)
        self.dense = DenseBlock(LAYERS)
        self.q = Q(2 * K * (LAYERS + 1))

    def forward(self, data):
        x = data.x
        x = self.input(x)
        x = self.dense(x, data.edge_index)
        actions = x[data.action_index]
        return self.q(actions)
