import numpy as np
from games import x2048
from agents import DQNagent

dim = 2
n = 20
eps = 10

hyparams = {
    'learning rate': 0.01,
    'batch size': 32,
    'eps settings': (1.0, 0.01, 0.995),
    'gamma': 0.8,
    'memory length': 2000
}

game = x2048(dim)
agent = DQNagent(game, hyparams)
agent.disableGPU()
agent.train()
