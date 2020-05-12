import atp
import clauses

from concurrent.futures import ThreadPoolExecutor
import math

ALPHA = 0.99

def select_inference(axioms, selected, inference):
    new_axioms = [axiom for axiom in axioms if axiom != inference]
    new_selected = selected + [inference]
    return new_axioms, new_selected

def uct(parent, child):
    if child.closed:
        return float('-inf')
    return child.score + math.sqrt(math.log(parent.visits) / child.visits)

class Node:
    executor = ThreadPoolExecutor(4)

    def __init__(self, baseline, axioms, selected, extras):
        self.baseline = baseline
        self.visits = 1
        self.closed = False
        self.children = None
        try:
            self.raw_score = self.baseline - atp.score(axioms, selected, extras)
            inferred, extras = atp.infer(selected, extras)
            self.inferences = axioms + inferred
            self.extras = extras
        except (atp.Crashed, atp.Timeout) as _:
            self.raw_score = -baseline
            self.closed = True
        except atp.ProvedIt:
            self.raw_score = self.baseline
            self.closed = True

    @property
    def score(self):
        return self.raw_score / self.baseline

    def select_child(self):
        assert not self.closed
        assert self.children is not None
        ucts = [uct(self, child) for child in self.children]
        argmax = max(enumerate(ucts), key=lambda x: x[1])[0]
        inference = self.inferences[argmax]
        child = self.children[argmax]
        return inference, child

    def expand(self, axioms, selected):
        assert not self.closed
        def new_child(inference):
            new_axioms, new_selected = select_inference(
                axioms,
                selected,
                inference
            )
            return Node(self.baseline, new_axioms, new_selected, self.extras)
        self.children = list(Node.executor.map(new_child, self.inferences))
        if not self.children:
            self.closed = True

    def update(self):
        assert self.children is not None
        self.raw_score = ALPHA * max(
            child.raw_score
            for child in self.children
        )
        self.closed = all(child.closed for child in self.children)

    def step(self, axioms, selected):
        self.visits += 1
        if self.children is None:
            self.expand(axioms, selected)
        else:
            inference, child = self.select_child()
            axioms, selected = select_inference(axioms, selected, inference)
            child.step(axioms, selected)
        self.update()

    def save_graphs(self, save_to, axioms, selected):
        import graphs
        if self.children is None or self.children == []:
            return
        nodes, sources, targets, indices = clauses.graph(
            [atp.tptp_clause(*clause) for clause in selected],
            [atp.tptp_clause(*clause) for clause in self.inferences],
        )
        y = [child.score for child in self.children]
        graphs.save(save_to, nodes, sources, targets, indices, y)
        for inference, child in zip(self.inferences, self.children):
            child_axioms, child_selected = select_inference(
                axioms,
                selected,
                inference
            )
            child.save_graphs(save_to, child_axioms, child_selected)
