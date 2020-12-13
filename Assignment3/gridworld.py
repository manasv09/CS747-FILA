'''
Author: Manas Vashistha
'''
import numpy as np

class GridWorld:

    def __init__(self, n_cols, n_rows, wind, n_actions, start, rew, stochastic):
        self.cols = n_cols
        self.rows = n_rows
        self.n_states = self.cols * self.rows
        self.wind = wind
        self.n_actions = n_actions
        assert self.n_actions == 4 or self.n_actions == 8
        self.start = start
        self.rew = rew
        self.start_st = self.ctos(self.start[0], self.start[1])
        self.rew_st = self.ctos(self.rew[0], self.rew[1])
        self.stochastic = stochastic

    def ctos(self, x, y):
        return y + x * self.rows
    
    def stoc(self, s):
        t = (s // self.rows)
        return t, s - t * self.rows
    
    def options(self, s):
        if self.n_actions == 4:
            return range(4)
        
        if self.n_actions == 8:
            return range(8)
    
    def move(self, s, a):
        x, y = self.stoc(s)
        y -= self.wind[x]

        if self.n_actions == 4:
            if a == 0:
                y -= 1
            if a == 1:
                x += 1
            if a == 2:
                y += 1
            if a == 3:
                x -= 1
        
        if self.n_actions == 8:
            if a == 0 or a == 1 or a == 7:
                y -= 1
            if a == 1 or a == 2 or a == 3:
                x += 1
            if a == 3 or a == 4 or a == 5:
                y += 1
            if a == 5 or a == 6 or a == 7:
                x -= 1
         
        if self.stochastic:
            if np.random.uniform() < 1/3:
                y -= 1
            elif np.random.uniform() < 2/3:
                y += 1
        
        x = max(min(self.cols - 1, x), 0)
        y = max(min(self.rows - 1, y), 0)

        return self.ctos(x, y), -1
    

    



    
