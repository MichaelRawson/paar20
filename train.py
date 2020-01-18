import gc
from glob import glob
import random

import torch
from torch.nn.functional import smooth_l1_loss
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from atp import Crashed, Timeout, ProvedIt
from environment import Environment
from model import Model
from replay import Replay

EPISODE_LENGTH = 10
GAMMA = 0.99
MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.9995
REPLAY_SIZE = 50000
MIN_REPLAY = 256
TARGET_UPDATE = 100

MINIBATCH = 32
LR = 1e-3
ALPHA = 0.95

def preprocess(env):
    nodes, sources, models, clauses = env.transform()
    x = torch.eye(11, dtype=torch.float)[nodes]
    edge_index = add_self_loops(
        torch.tensor([sources, models], dtype=torch.long),
        num_nodes = len(nodes)
    )[0]
    action_index = torch.tensor(clauses[-len(env.actions):], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.action_index = action_index
    return data

def episode(model, path, epsilon):
    try:
        env = Environment(path)
    except (Timeout, ProvedIt, Crashed):
        print(f"{path}: skipped")
        return []

    data = preprocess(env)
    terminal = False
    total = 0
    transitions = []

    for t in range(EPISODE_LENGTH):
        # shouldn't happen, very occasionally does (??)
        if len(env.actions) == 0:
            break

        if random.random() < epsilon:
            action = random.randint(0, len(env.actions) - 1)
        else:
            with torch.no_grad():
                action = model(data.clone().to('cuda')).argmax().item()

        try:
            reward = env.perform_action(action)
        except Timeout:
            reward = -1 - total
            terminal = True
        except ProvedIt:
            reward = 1 - total
            terminal = True
        except Crashed:
            print(f"{path}: crashed")
            return []

        total += reward
        if total + reward <= -1.0 or t == EPISODE_LENGTH - 1:
            terminal = True

        if terminal:
            next_data = None
        else:
            next_data = preprocess(env)

        transition = (data, action, reward, next_data)
        transitions.append(transition)
        data = next_data
        if terminal:
            break

    print(f"{path}: OK")
    return transitions

def main():
    paths = ['Problems/GRP/GRP001-1.p'] #glob('m2k/*')

    model = Model().to('cuda')
    target = Model().to('cuda')
    epsilon = MAX_EPSILON
    optimizer = RMSprop(model.parameters(), lr=LR, alpha=ALPHA)
    replay = Replay(REPLAY_SIZE)
    writer = SummaryWriter()

    episodes = 0
    examples = 0
    parameter_updates = 0
    while True:
        path = random.choice(paths)
        transitions = episode(model, path, epsilon)
        for transition in transitions:
            replay.push(transition)

        if len(transitions) > 0:
            episodes += 1
            _, _, rewards, _ = zip(*transitions)
            writer.add_scalar('reward', sum(rewards), episodes)

        if len(replay) < MIN_REPLAY:
            continue

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                writer.add_histogram(
                    name.replace('.', '/'),
                    parameter.data,
                    parameter_updates
                )

        optimizer.zero_grad()
        for _ in range(MINIBATCH):
            data, action, reward, next_data = replay.sample()
            y = torch.tensor(reward, dtype=torch.float)
            if next_data is not None and parameter_updates > TARGET_UPDATE:
                with torch.no_grad():
                    y += GAMMA * target(next_data.clone().to('cuda')).max()

            data = data.clone().to('cuda')
            q = model(data)
            loss = smooth_l1_loss(q[action], y.to('cuda'))
            loss.backward()
            examples += 1
            writer.add_histogram('Q', q, examples)
            writer.add_scalar('loss', loss, examples)

        optimizer.step()
        parameter_updates += 1
        if parameter_updates % TARGET_UPDATE == 0:
            torch.save(model.state_dict(), 'model.pt')
            target.load_state_dict(model.state_dict())

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        gc.collect()

if __name__ == '__main__':
    main()
