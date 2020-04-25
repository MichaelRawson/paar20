import atp
import clauses
from data import normalised_adjacencies
from model import Model

import math
from queue import PriorityQueue
import torch

def select_inference(axioms, selected, inference):
    new_axioms = [axiom for axiom in axioms if axiom != inference]
    new_selected = selected + [inference]
    return new_axioms, new_selected

def mk_inputs(selected, inferences):
    nodes, sources, targets, indices = clauses.graph(
        [atp.tptp_clause(*clause) for clause in selected],
        [atp.tptp_clause(*clause) for clause in inferences],
    )
    nodes = torch.tensor(nodes)
    sources = torch.tensor(sources)
    targets = torch.tensor(targets)
    indices = torch.tensor(indices)
    edge_index, edge_weight_t, edge_index_t, edge_weight = normalised_adjacencies(nodes, sources, targets)
    adjacency = torch.sparse.FloatTensor(edge_index, edge_weight)
    adjacency_t = torch.sparse.FloatTensor(edge_index_t, edge_weight_t)
    assignment = torch.zeros(nodes.shape)
    return (1, assignment.to('cuda'), nodes.to('cuda'), adjacency.to('cuda'), adjacency_t.to('cuda'), indices.to('cuda'))

if __name__ == '__main__':
    from pathlib import Path
    import sys
    path = Path(sys.argv[1])
    model = Model().to('cuda')
    model.load_state_dict(torch.load('data/GRP001-1.pt'))
    model.eval()
    axioms, selected = atp.clausify(path)
    queue = PriorityQueue()
    queue.put((-1.0, axioms, selected))
    while not queue.empty():
        negative_prob, axioms, selected = queue.get()
        prob = -negative_prob
        inferences = axioms + atp.infer(selected)
        with torch.no_grad():
            logits = model(*mk_inputs(selected, inferences))
        negative_probs = negative_prob * torch.softmax(logits, dim=0)
        for negative_prob, inference in zip(negative_probs, inferences):
            new_axioms, new_selected = select_inference(axioms, selected, inference)
            queue.put((negative_prob, new_axioms, new_selected))
