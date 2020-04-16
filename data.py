import torch
from uuid import uuid4

def normalised_adjacencies(nodes, sources, targets):
    weights = torch.ones(sources.shape)
    degree = torch.zeros(nodes.shape).scatter_add(0, sources, weights)
    normalise = 1.0 / degree
    edge_index = torch.stack([sources, targets])
    edge_weight = normalise[sources]

    degree_t = torch.zeros(nodes.shape).scatter_add(0, targets, weights)
    normalise_t = 1.0 / degree_t
    edge_index_t = torch.stack([targets, sources])
    edge_weight_t = normalise_t[targets]
    return edge_index, edge_weight, edge_index_t, edge_weight_t

def save(save_to, nodes, sources, targets, indices, y):
    assert len(sources) == len(targets)
    nodes = torch.tensor(nodes)
    sources = torch.tensor(sources)
    targets = torch.tensor(targets)
    indices = torch.tensor(indices)
    edge_index, edge_weight, edge_index_t, edge_weight_t =\
        normalised_adjacencies(nodes, sources, targets)
    y = torch.tensor(y)
    record = {
        'nodes': nodes,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'edge_index_t': edge_index_t,
        'edge_weight_t': edge_weight_t,
        'indices': indices,
        'y': y
    }
    uuid = uuid4().hex
    torch.save(record, f'{save_to}/{uuid}.pt')
