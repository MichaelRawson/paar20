from uuid import uuid4
from glob import glob
import random

import torch
from torch.utils.data import DataLoader, IterableDataset

# compute the incoming-normalised adjacency and transpose-adjacency matrices
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

def graphs(pat):
    paths = glob(pat)
    random.shuffle(paths)
    for path in paths:
        record = torch.load(path)
        yield record

def batches(pat):
    for record in graphs(pat):
        nodes = record['nodes'].to('cuda')
        edge_index = record['edge_index'].to('cuda')
        edge_index_t = record['edge_index_t'].to('cuda')
        edge_weight = record['edge_weight'].to('cuda')
        edge_weight_t = record['edge_weight_t'].to('cuda')
        indices = record['indices'].to('cuda')
        y = record['y'].to('cuda')
        adjacency = torch.sparse.FloatTensor(edge_index, edge_weight)
        adjacency_t = torch.sparse.FloatTensor(edge_index_t, edge_weight_t)
        yield (nodes, adjacency, adjacency_t, indices, y)

def loader(pat):
    return batches(pat)
