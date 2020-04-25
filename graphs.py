from uuid import uuid4
from glob import glob
import random

import torch
from torch.utils.data import DataLoader, IterableDataset

BATCH_NODES = 10000
TEMPERATURE = 1

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

def collate(
    total,
    batch_num_examples,
    assignment_batch,
    node_batch,
    edge_index_batch,
    edge_index_t_batch,
    edge_weight_batch,
    edge_weight_t_batch,
    index_batch,
    y_batch
):
    num_examples = batch_num_examples
    assignment = torch.cat(assignment_batch, dim=0)
    nodes = torch.cat(node_batch, dim=0).to('cuda')
    edge_index = torch.cat(edge_index_batch, dim=1).to('cuda')
    edge_index_t = torch.cat(edge_index_t_batch, dim=1).to('cuda')
    edge_weight = torch.cat(edge_weight_batch, dim=0).to('cuda')
    edge_weight_t = torch.cat(edge_weight_t_batch, dim=0).to('cuda')
    indices = torch.cat(index_batch, dim=0).to('cuda')
    y = torch.cat(y_batch, dim=0).to('cuda')
    adjacency = torch.sparse.FloatTensor(edge_index, edge_weight)
    adjacency_t = torch.sparse.FloatTensor(edge_index_t, edge_weight_t)
    assignment_batch = []
    node_batch = []
    edge_index_batch = []
    edge_index_t_batch = []
    edge_weight_batch = []
    edge_weight_t_batch = []
    index_batch = []
    y_batch = []
    total = 0
    batch_num_examples = 0
    return (
        num_examples,
        assignment,
        nodes,
        adjacency,
        adjacency_t,
        indices,
        y
    )

def batches(pat):
    total = 0
    batch_num_examples = 0
    assignment_batch = []
    node_batch = []
    edge_index_batch = []
    edge_index_t_batch = []
    edge_weight_batch = []
    edge_weight_t_batch = []
    index_batch = []
    y_batch = []

    for record in graphs(pat):
        nodes = record['nodes']
        if total > 0 and total + len(nodes) > BATCH_NODES:
            yield collate(
                total,
                batch_num_examples,
                assignment_batch,
                node_batch,
                edge_index_batch,
                edge_index_t_batch,
                edge_weight_batch,
                edge_weight_t_batch,
                index_batch,
                y_batch
            )

        assignment_batch.append(torch.full(
            (len(nodes),), len(node_batch),
            dtype=torch.long
        ))
        edge_index_batch.append(record['edge_index'] + total)
        edge_index_t_batch.append(record['edge_index_t'] + total)
        edge_weight_batch.append(record['edge_weight'])
        edge_weight_t_batch.append(record['edge_weight_t'])
        index_batch.append(record['indices'] + total)
        y_batch.append(torch.softmax(TEMPERATURE * record['y'], dim=0))
        node_batch.append(nodes)
        total += len(nodes)
        batch_num_examples += 1

    if total > 0:
        yield collate(
            total,
            batch_num_examples,
            assignment_batch,
            node_batch,
            edge_index_batch,
            edge_index_t_batch,
            edge_weight_batch,
            edge_weight_t_batch,
            index_batch,
            y_batch
        )

def loader(pat):
    return batches(pat)
