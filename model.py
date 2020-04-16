import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import log_softmax, relu
from torch.nn.init import xavier_normal_

NODE_TYPES = 11
CHANNELS = 32
HIDDEN_LAYER = 1024
RESIDUAL_LAYERS = 8

class Conv(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.Tensor(CHANNELS, CHANNELS // 2))
        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.weight)

    def forward(self, nodes, adjacency):
        return adjacency @ nodes @ self.weight

class BiConv(Module):
    def __init__(self):
        super().__init__()
        self.out = Conv()
        self.back = Conv()

    def forward(self, nodes, adjacency, adjacency_t):
        out = self.out(nodes, adjacency)
        back = self.back(nodes, adjacency_t)
        nodes = torch.cat([out, back], dim=1)
        return nodes

class BiConvRes(Module):
    def __init__(self):
        super().__init__()
        self.bn1 = BatchNorm1d(CHANNELS)
        self.conv1 = BiConv()
        self.bn2 = BatchNorm1d(CHANNELS)
        self.conv2 = BiConv()

    def forward(self, skip, adjacency, adjacency_t):
        nodes = skip
        nodes = relu(self.bn1(nodes))
        nodes = self.conv1(nodes, adjacency, adjacency_t)
        nodes = relu(self.bn2(nodes))
        nodes = self.conv2(nodes, adjacency, adjacency_t)
        return skip + nodes

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv0 = BiConv()
        self.res = ModuleList([BiConvRes() for _ in range(RESIDUAL_LAYERS)])
        self.hidden = Linear(CHANNELS, HIDDEN_LAYER)
        self.output = Linear(HIDDEN_LAYER, 1)

    def forward(self, num_examples, assignment, nodes, adjacency, adjacency_t, indices):
        nodes = self.embedding(nodes)
        nodes = self.conv0(nodes, adjacency, adjacency_t)
        for res in self.res:
            nodes = res(nodes, adjacency, adjacency_t)

        def final(nodes):
            nodes = self.hidden(relu(nodes))
            nodes = self.output(relu(nodes)).squeeze()
            return log_softmax(nodes, dim=0)

        outputs = torch.cat([
            final(nodes[indices[assignment[indices] == item]])
            for item in range(num_examples)
        ], dim=0)
        return outputs
