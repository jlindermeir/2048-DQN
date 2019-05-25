import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from games import x2048
from agents import DQNagent


dim = 2
episodes0 = 5000

hyparams = (
    #'learning rate':
    0.01,
    #'batch size':
    32,
    #'eps settings':
    (1.0, 0.01, 0.995),
    #'gamma':
    0.8,
    #'memory length':
    2000
)

game = x2048(dim)
arrs = []
for paramset in [hyperparams]:
    agent = DQNagent(game, hyparams)
    agent.disableGPU()
    sarr = agent.Train(episodes = 5000)[0]
    sarr_filtered = gaussian_filter(sarr, 100)

plt.figure(1)
