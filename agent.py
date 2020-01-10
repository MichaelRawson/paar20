from environment import Environment, Timeout, ProvedIt

import random

class Agent:
    def __init__(self, environment, policy, epsilon=0.01):
        self.environment = environment
        self.policy = policy
        self.epsilon = epsilon

    def episode(self, length=10):
        chosen = []
        rewards = []
        total = 0

        for _ in range(length):
            choice = self.policy(self.environment).argmax().item()
            if random.random() < self.epsilon:
                choice = random.randint(0, len(self.environment.actions) - 1)

            chosen.append(choice)
            try:
                rewards.append(self.environment.perform_action(choice))
            except Timeout:
                rewards.append(-1)
                break
            except ProvedIt:
                rewards.append(1)
                break

            total += rewards[-1]

            if total <= -1.0:
                break

        return (chosen, rewards)
