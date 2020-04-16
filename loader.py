from glob import glob
import random
import torch
from torch.utils.data import DataLoader, IterableDataset

BATCH_NODES = 10000
TEMPERATURE = 10

def graphs(pat):
    paths = glob(pat)
    random.shuffle(paths)
    for path in paths:
        record = torch.load(path)
        yield record

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

    def collate():
        nonlocal total, batch_num_examples, assignment_batch, node_batch, edge_index_batch, edge_index_t_batch, edge_weight_batch, edge_weight_t_batch, index_batch, y_batch
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
        return (num_examples, assignment, nodes, adjacency, adjacency_t, indices, y)

    for record in graphs(pat):
        nodes = record['nodes']
        if total > 0 and total + len(nodes) > BATCH_NODES:
            yield collate()

        assignment_batch.append(torch.full((len(nodes),), len(node_batch), dtype=torch.long))
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
        yield collate()

def loader(pat):
    return batches(pat)
