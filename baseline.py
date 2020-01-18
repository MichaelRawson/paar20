from concurrent.futures import ThreadPoolExecutor
from glob import glob
import random

from agent import Agent
from environment import Environment
from model import Model
from atp import VampireCrashed, Timeout, ProvedIt

import torch

def random_policy(env):
    policy = [0.0] * len(env.actions)
    policy[random.randint(0, len(env.actions) - 1)] = 1.0
    return torch.tensor(policy)

def episode(path):
    try:
        env = Environment(path)
        agent = Agent(env, random_policy)
        chosen, rewards = agent.episode()
        return rewards
    except (Timeout, ProvedIt, VampireCrashed):
        pass

def baseline():
    paths = glob('m30k/*') + glob('m2k/*')
    for reward in ThreadPoolExecutor(4).map(
        episode,
        [random.choice(paths) for _ in range(256)]
    ):
        if reward is not None:
            print(sum(reward))

if __name__ == '__main__':
    baseline()
