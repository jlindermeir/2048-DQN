import numpy as np
import os
from time import time
import random
import readchar
from collections import deque
from keras.models import Sequential, load_model
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
        self.memory = deque(maxlen=int(hyparams[6]))

        self.learning_rate = max(hyparams[0], 0)
        self.batch_size = int(hyparams[1])
        self.epsilon = min(hyparams[2], 1) # exploration rate
        self.epsilon_min = max(hyparams[3], 0)
        self.epsilon_decay = min(hyparams[4], 0.999)
        self.gamma = hyparams[5]

        if model:
            self.model = model
        else:
            self.model = self.cModel4x4()

    def cModel(self, file=None):
        if file:
            return load_model(file)
        else:
            model = Sequential()
            model.add(Dense(24, input_dim=self.dim**2, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(4, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def cModel4x4(self, file=None):
        if file:
            return load_model(file)
        else:
            model = Sequential()
            model.add(Dense(48, input_dim=self.dim**2, activation='relu'))
            model.add(Dense(48, activation='relu'))
            model.add(Dense(48, activation='relu'))
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

    def replay(self):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, self.batch_size)
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
        self.model.fit(xarr,yarr, batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def iTrain(self):
        avgdur = 100
        earr = []
        scorearr = []
        turnarr = []
        cscorearr = []
        i = 1

        while True:
            try:
                c_arrs = self.train()

                scorearr += c_arrs[0]
                cscorearr.append(sum(scorearr))
                earr.append(self.epsilon)

                if not i % avgdur:
                    avgarr = scorearr[-avgdur:-1]
                    print("GAME {} SCORE[median:{} mean:{}, stddev:{:.2}], e: {:.2}".format(i, int(np.median(avgarr)), np.mean(avgarr), np.std(avgarr), self.epsilon))
                i+=1

            except KeyboardInterrupt:
                inpt = readchar.readchar()
                if inpt == 'p':
                    self.game.__init__(self.dim)
                    self.game.play(self.predict, wfu=True)
                    self.game.__init__(self.dim)
                elif inpt == 'g':
                    self.plot([scorearr, cscorearr, earr], sigma=avgdur)
                elif inpt == 'q':
                    print()
                    break
                print("\nResuming training...")

    def train(self, episodes=1, savedir=None, pts = False):
        scorearr = []
        turnarr = []

        time0 = time()
        for i in range(episodes):
            if pts: print("Training episode %i/%i" % (i, episodes), end='\r')
            self.game.__init__(self.dim)
            while not self.game.lost:
                oldscore = self.game.score
                oldstate = self.game.arr
                action = self.predict(self.game.arr)
                self.game.move(action)
                reward = self.game.score - oldscore
                self.remember(oldstate, action, reward, self.game.arr, self.game.lost)

            scorearr.append(float(self.game.score))
            turnarr.append(float(self.game.turns))

            if savedir: self.model.save('%s/eps%i.h5' % (savedir, i))
            if len(self.memory) > self.batch_size:
                self.replay()
        time1 = time()
        avgtime = (time1/time0)/episodes
        if pts: print('Training done after %f s (%f per ep).' % (time1 - time0, avgtime),' '*20)
        return scorearr, turnarr, avgtime


    def plot(self, arrlist, sigma=None):
        plt.figure(1)
        for i,arr in enumerate(arrlist):
            plt.subplot(1e2 * len(arrlist) + 10 + (i+1))
            if sigma and i == 0:
                plt.plot(arr, 'k+')
                plt.plot(gaussian_filter(arr, sigma, mode='reflect'), 'r')
            else:
                plt.plot(arr, 'r')
        plt.show()

    def disableGPU(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def eval(self, n_games, pts=False):
        scorearr = []
        turnarr = []
        maxtile = 0

        time0 = time()
        for i in range(n_games):
            if pts: print("Evaluating, game %i/%i..." % (i, n_games), end='\r')
            score, turns = self.game.play(self.predict, pts=False)
            scorearr.append(score)
            turnarr.append(turns)
            maxtile = max(np.max(self.game.arr), maxtile)
        score_mean = np.mean(scorearr)
        score_err = np.std(scorearr)
        turns_mean = np.mean(turnarr)
        turns_err = np.std(turnarr)
        time1 = time()
        avgtime = (time1 - time0)/n_games
        if pts: print('Evaluation done after %f s (%f per game).' % (time1 - time0, avgtime), ' '*20)
        return score_mean, score_err, maxtile, turns_mean, turns_err
