import numpy as np
import os
from time import time
import random
import readchar
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def human(arr):
    inptdict = {4:0, 8:1, 6:2, 2:3}
    inpt = readchar.readchar()
    return inptdict[int(inpt)]

def greedy(arr):
    tgame = game(arr.shape[0])
    score_list = []
    for i in range(4):
        tgame.arr = arr
        tgame.score = 0
        tgame.move(i)
        score_list.append(tgame.score)
    if max(score_list) == 0:
        return np.random.randint(0,4)
    else:
        return score_list.index(max(score_list))

def rand(arr):
    return np.random.randint(0,4)

class DQNagent:
    def __init__(self, game, hyparams, model=None):
        self.dim = game.dim
        self.game = game
        self.memory = deque(maxlen=hyparams['memory length'])

        self.gamma = hyparams['gamma']
        self.epsilon = hyparams['eps settings'][0] # exploration rate
        self.epsilon_min = hyparams['eps settings'][1]
        self.epsilon_decay = hyparams['eps settings'][2]
        self.learning_rate = hyparams['learning rate']
        self.batch_size = hyparams['batch size']

        if model:
            self.model = model
        else:
            self.model = self.cModel(self.dim**2)

    def cModel(self, inptDim):
        model = Sequential()
        model.add(Dense(24, input_dim=inptDim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def pad(self,arr):
        return np.reshape(arr.flatten(), [1, self.dim**2])

    def predict(self, state):
        if np.random.rand() <= self.epsilon:
            action = rand(state)
            while not self.checkTurn(state, action):
                action = rand(state)
        else:
            actionVals = self.model.predict(self.pad(state))[0]
            action = np.argmax(actionVals)
            while not self.checkTurn(state, action):
                actionVals[action] = float("-inf")
                action = np.argmax(actionVals)
        return action

    def checkTurn(self, arr, dir):
        arr = np.rot90(arr, dir)
        for row in arr:
            zero = False
            for i,val in enumerate(row[0:-1]):
                zero = zero or val == 0
                if row[i+1]!=0 and (zero or row[i]==row[i+1]): return True
        return False


    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            elif np.array_equal(state, next_state):
                target = 0
            else:
                target = reward + self.gamma * np.amax(self.model.predict(self.pad(next_state))[0])
            target_f = self.model.predict(self.pad(state))
            target_f[0][action] = target
            x_batch.append(state.flatten())
            y_batch.append(target_f.flatten())
        xarr = np.stack(x_batch)
        yarr = np.stack(y_batch)
        #print(xarr.shape, yarr.shape)
        self.model.fit(xarr,yarr, batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, batch_size=32, episodes=None):
        avgdur = 100
        scorearr = []
        turnarr = []
        mtarr = [] #maximum tile
        earr = []
        i = 1

        counter = time()
        while True:
            try:
                self.game.__init__(self.dim)
                while not self.game.lost:
                    oldscore = self.game.score
                    oldstate = self.game.arr

                    action = self.predict(self.game.arr)
                    self.game.move(action)
                    reward = self.game.score - oldscore

                    self.remember(oldstate, action, reward, self.game.arr, self.game.lost)

                scorearr.append(self.game.score)
                turnarr.append(self.game.turns)
                mtarr.append(np.amax(self.game.arr))
                earr.append(self.epsilon)

                if not i % avgdur:
                    gametime = (time() - counter)/avgdur
                    avgarr = scorearr[-avgdur:-1]
                    print("GAME {} SCORE[median:{} mean:{}, stddev:{:.2}], e: {:.2}, time per game {}".format(i, int(np.median(avgarr)), np.mean(avgarr), np.std(avgarr), self.epsilon, gametime))
                    #if self.sPlot: self.plot([scorearr, mtarr, earr], sigma=avgdur)
                    counter = time()

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

                if episodes and i==episodes:
                    break
                i+=1
            except KeyboardInterrupt:
                inpt = readchar.readchar()
                if inpt == 'p':
                    tgame = game(self.dim)
                    tgame.play(self.predict, wfu=True)
                elif inpt == 'g':
                    self.sPlot = True
                    self.plot([scorearr, mtarr, earr], sigma=avgdur)
                elif inpt == 'q':
                    print()
                    break
                print("\nResuming training...")
        return scorearr, turnarr, mtarr, earr

    def plot(self, arrlist, sigma=None):
        plt.figure(1)
        #plt.clf()
        for i,arr in enumerate(arrlist):
            plt.subplot(1e2 * len(arrlist) + 10 + (i+1))
            if sigma and (i+1) != len(arrlist):
                plt.plot(arr, 'k+')
                plt.plot(gaussian_filter(arr, sigma, mode='reflect'), 'r')
            else:
                plt.plot(arr, 'r')
        plt.show()

    def disableGPU(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
