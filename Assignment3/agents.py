'''
Author: Manas Vashistha
'''
import numpy as np

class Agents:

    def __init__(self, alpha, gw, gamma = 1):
        self.alpha = alpha
        self.gw = gw
        self.gamma = gamma
        self.Q = np.zeros((self.gw.n_states, self.gw.n_actions))
        self.s0 = self.gw.start_st
        self.sn = self.gw.rew_st
        self.st = self.s0
        
    def execute(self, eps):
        self.st = self.s0
        iteration = 0
        at = self.pi(self.Q, self.st, eps)
        while(self.st != self.sn):
            st1, rt1 = self.gw.move(self.st, at)
            at1 = self.pi(self.Q, st1, eps)
            self.Q[self.st, at] += self.alpha * (self.target(rt1, st1, at1, eps) - self.Q[self.st, at])
            self.st = st1
            at = at1
            iteration += 1
        return iteration

    def pi(self, Q, s, eps):
        opt = self.gw.options(s)

        if np.random.uniform() < eps:
            return np.random.choice(opt)
        else:
            return np.random.choice(np.where(Q[s] == Q[s].max())[0])
    
    def target(self, r, s, a, eps):
        raise NotImplementedError('Agents is an Abstract Class')


class SARSA(Agents):

    def __init__(self, alpha, gw, gamma=1):
        super().__init__(alpha, gw, gamma)
    
    def target(self, r, s, a, eps):
        return r + self.gamma * self.Q[s, a]


class ExpSARSA(Agents):

    def __init__(self, alpha, gw, gamma=1):
        super().__init__(alpha, gw, gamma)

    def target(self, r, s, a, eps):
        t = eps * self.Q[s].mean() + (1-eps) * self.Q[s].max()
        return r + self.gamma * t

class QLearning(Agents):

    def __init__(self, alpha, gw, gamma=1):
        super().__init__(alpha, gw, gamma)

    def target(self, r, s, a, eps):
        return r + self.gamma * self.Q[s].max()
        