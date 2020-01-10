import subprocess

import clauses
import vampire

class ProvedIt(Exception):
    pass

class Timeout(Exception):
    pass

class Environment:
    def __init__(self, problem):
        self.problem = problem
        clauses = vampire.clausify(problem)

        self.actions = [
            clause for clause in clauses
            if b',negated_conjecture,' not in clause
        ]
        self.selected = [
            clause for clause in clauses
            if b',negated_conjecture,' in clause
        ]
        self.axioms_available = len(self.actions)
        self._update_actions()
        self._update_score()
        self.initial = self.score

    def _update_actions(self):
        inferred = vampire.infer(self.selected)
        if inferred is None:
            raise ProvedIt()
        self.actions = self.actions[:self.axioms_available] + inferred

    def _update_score(self):
        try:
            self.score = vampire.score(
                self.actions[:self.axioms_available] + self.selected
            )
        except subprocess.TimeoutExpired:
            raise Timeout()

    def transform(self):
    	return clauses.graph(self.selected, self.actions)

    def perform_action(self, index):
        self.selected.append(self.actions[index])
        del self.actions[index]
        if index < self.axioms_available:
            self.axioms_available -= 1
        self._update_actions()

        old = self.score
        self._update_score()
        return (old - self.score) / self.initial
