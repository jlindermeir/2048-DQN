import numpy as np
from games import x2048
from agents import DQNagent

dim = 2
n = 5
eps = 400

game = x2048(dim)
agent = DQNagent(game)
agent.disableGPU()
slist = []


agent.model.save_weights('model.h5')
for i in range(n):
    agent.model.load_weights('model.h5')
    agent.epsilon = 1
    slist.append(agent.train(episodes = eps)[0])
    print('%i done' % i)

slist.append([0,0])
agent.plot(slist, sigma=100)
