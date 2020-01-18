import subprocess

import atp
import clauses

class Environment:
    def __init__(self, problem):
        self.problem = problem
        clauses = atp.clausify(problem)

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
        inferred = atp.infer(self.selected)
        self.actions = self.actions[:self.axioms_available] + inferred

    def _update_score(self):
        self.score = atp.score(
            self.actions[:self.axioms_available] + self.selected
        )

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
