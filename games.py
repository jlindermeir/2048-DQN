import numpy as np
import os
from time import time
import random
import readchar

class x2048:
    def __init__(self, dim):
        self.dim = dim
        self.turns = 0
        self.score = 0
        self.lost = False

        self.arr = np.zeros((dim,dim), dtype=np.int16)
        self.add2()
        self.add2()

    def add2(self):
        zrs = np.argwhere(self.arr == 0)
        k = np.random.randint(0, len(zrs))
        self.arr[zrs[k][0],zrs[k][1]] = 2

    def move(self, direction):
        #0:left, 1:up, 2:right, 3:down
        def c_row(row):
            ints = np.argwhere(row > 0)
            c_row = np.array([row[i] for i in ints], dtype=np.int16)
            zrs = np.zeros(self.dim - c_row.shape[0], dtype=np.int16)
            return np.append(c_row, zrs)

        rowarr = []
        for row in np.rot90(self.arr, direction):
            row = c_row(row)
            for i in range(self.dim - 1):
                if row[i]==row[i+1]:
                    row[i] *= 2
                    row[i+1] = 0
                    self.score += row[i]
            row = c_row(row)
            rowarr.append(row)
        narr = np.rot90(np.stack(rowarr), -direction)

        if not np.array_equal(self.arr, narr):
            self.arr = narr
            self.turns += 1
            self.add2()

        if 0 not in self.arr:
            movepos = False
            for i in range(2):
                narr = np.rot90(self.arr, i)
                zrs = np.zeros((self.dim, 1), dtype=np.int16)
                a = np.concatenate([narr, zrs],1)
                b = np.concatenate([zrs, narr], 1)
                movepos |= np.amax((a - b) == 0)
                if movepos: break
            else:
                self.lost = True

    def showState(self, cls=True):
        if cls: os.system('clear')
        print('Score: %i' % self.score)
        print('Turns: %i' % self.turns)
        if self.lost:
            print('GAME OVER!')
        else:
            print()
        print(self.arr)

    def play(self, agent, pts=True, wfu=False):
        if pts: self.showState()
        while not self.lost:
            move = agent(self.arr)
            if wfu:
                print("Next Move: %i" % move)
                if 'q' == readchar.readchar(): break
            self.move(move)
            if pts: self.showState()
