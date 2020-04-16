from loader import loader
from model import Model

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

def epoch(writer, model, optimizer, steps):
    for batch in loader('data/GRP001-1/*.pt'):
        optimizer.zero_grad()
        num_examples, assignment, nodes, adjacency, adjacency_t, indices, y = batch
        log_predictions = model(
            num_examples,
            assignment,
            nodes,
            adjacency,
            adjacency_t,
            indices
        )
        error = -(y * log_predictions).mean()
        loss = num_examples * error
        loss.backward()
        optimizer.step()

        steps += 1
        writer.add_scalar('loss/X-entropy', error, steps)
        if steps % 100 == 0:
            writer.add_histogram('predicted', log_predictions.exp(), steps)
            writer.add_histogram('actual', y, steps)
            writer.add_histogram('error', log_predictions.exp() - y, steps)
    return steps

def train():
    writer = SummaryWriter()
    model = Model().to('cuda')
    optimizer = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    steps = 0
    while True:
        steps = epoch(writer, model, optimizer, steps)
        torch.save(model.state_dict(), 'data/model.pt')

if __name__ == '__main__':
    train()
