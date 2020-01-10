from concurrent.futures import ThreadPoolExecutor
from glob import glob
import random

import torch
from torch.nn.functional import softmax, log_softmax, smooth_l1_loss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

from agent import Agent
from environment import Environment, Timeout, ProvedIt
from model import Model
from vampire import VampireCrashed

def graph_data(state):
    nodes, sources, targets, clauses = state
    x = torch.eye(11, dtype=torch.float)[nodes]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    clause_index = torch.tensor(clauses, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.clause_index = clause_index
    return data

class Policy:
    def __init__(self, model):
        self.model = model
        self.actor_log = []
        self.critic_log = []

    def __call__(self, env):
        data = graph_data(env.transform())
        data.action_index = data.clause_index[-len(env.actions):]
        data = data.to('cuda')
        policy, value = self.model(data)
        self.actor_log.append(policy)
        self.critic_log.append(value)
        return policy

def episode(model, path):
    print(path)

    try:
        env = Environment(path)
    except (Timeout, ProvedIt):
        return (None, None, None)

    policy = Policy(model)
    agent = Agent(env, policy)
    try:
        chosen, rewards = agent.episode()
    except VampireCrashed:
        return (None, None, None)

    policy(env)
    return (chosen, rewards, policy)

def train():
    paths = glob('m2k/*')
    model = Model().to('cuda')
    model.train()
    optimizer = SGD(model.parameters(), lr=1e-4)
    writer = SummaryWriter()

    iteration = 0
    while True:
        optimizer.zero_grad()
        for chosen, rewards, policy in ThreadPoolExecutor(4).map(
            lambda path: episode(model, path),
            [random.choice(paths) for _ in range(32)]
        ):
            if chosen is None:
                break

            actor_losses = []
            entropy_losses = []
            critic_losses = []

            ret = policy.critic_log[-1].detach()
            for t in range(len(rewards) - 1, -1, -1):
                ret += rewards[t]
                actor = policy.actor_log[t]
                critic = policy.critic_log[t]
                advantage = ret - critic.detach()

                actor_loss = -log_softmax(actor, dim=0)[chosen[t]] * advantage
                entropy_loss = (softmax(actor, dim=0) * log_softmax(actor, dim=0)).sum()
                critic_loss = smooth_l1_loss(critic, ret)
                loss = actor_loss + 0.01 * entropy_loss + 0.1 * critic_loss
                loss.backward(retain_graph=True)

                actor_losses.append(actor_loss)
                entropy_losses.append(entropy_loss)
                critic_losses.append(critic_loss)

            writer.add_scalar('reward', sum(rewards), iteration)
            writer.add_scalar('loss/actor', sum(actor_losses) / len(actor_losses), iteration)
            writer.add_scalar('loss/entropy', sum(entropy_losses) / len(entropy_losses), iteration)
            writer.add_scalar('loss/critic', sum(critic_losses) / len(critic_losses), iteration)
            iteration += 1

        optimizer.step()

if __name__ == '__main__':
    train()
