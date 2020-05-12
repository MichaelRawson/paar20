import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import softmax, relu
from torch.nn.init import xavier_normal_

NODE_TYPES = 11
CHANNELS = 64
HIDDEN_LAYER = 1024
RESIDUAL_LAYERS = 16

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
        nodes = self.bn1(relu(nodes))
        nodes = self.conv1(nodes, adjacency, adjacency_t)
        nodes = self.bn2(relu(nodes))
        nodes = self.conv2(nodes, adjacency, adjacency_t)
        return skip + nodes

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv0 = BiConv()
        self.res = ModuleList([BiConvRes() for _ in range(RESIDUAL_LAYERS)])
        self.hidden = Linear((RESIDUAL_LAYERS + 2) * CHANNELS, HIDDEN_LAYER)
        self.output = Linear(HIDDEN_LAYER, 1)

    def forward(self, nodes, adjacency, adjacency_t, indices):
        log = []
        nodes = self.embedding(nodes)
        log.append(nodes)
        nodes = self.conv0(nodes, adjacency, adjacency_t)
        log.append(nodes)
        for res in self.res:
             nodes = res(nodes, adjacency, adjacency_t)
             log.append(nodes)

        nodes = torch.cat(log, dim=1)
        nodes = nodes[indices]
        nodes = self.hidden(relu(nodes))
        nodes = self.output(relu(nodes)).squeeze()
        return softmax(nodes, dim=0)
