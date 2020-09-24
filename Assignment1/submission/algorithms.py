import numpy as np

class Algorithms:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_pulls = np.zeros(self.num_arms, dtype=int)
        self.pulls = 0
        self.cum_rew = 0
        self.rewards = np.zeros(self.num_arms)

    def execute(self, instance):
        self.at = self.run_algo()
        self.rt = instance.pull_arm(self.at)
        self.arm_pulls[self.at] += 1
        self.rewards[self.at] += self.rt
        self.pulls = np.sum(self.arm_pulls)
        self.cum_rew = np.sum(self.rewards)
    
    def run_algo(self):
        raise NotImplementedError
    
    def tie_breaker(self, x):
        return np.random.choice(np.where(x == x.max())[0])


class EpsilonGreedy(Algorithms):
    def __init__(self, epsilon, num_arms):
        super().__init__(num_arms)
        self.epsilon = epsilon

    def run_algo(self):
        if np.random.binomial(1, self.epsilon):
            return np.random.randint(self.num_arms) # explore with probability epsilon
        
        pt = np.zeros_like(self.rewards)
        idxs = (self.arm_pulls != 0)
        pt[idxs] = self.rewards[idxs] / self.arm_pulls[idxs]
        return self.tie_breaker(pt)
        
class UCB(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
    
    def run_algo(self):
        if self.pulls < self.num_arms:
            return self.pulls # sample all the arms exactly once while starting
        pt = self.rewards / self.arm_pulls
        ucbt = pt + np.sqrt(2 * np.log(self.pulls) / self.arm_pulls)
        return self.tie_breaker(ucbt)

class KLUCB(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
        self.c = 3
        
    def kld(self, x, y):
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
        
    def bin_search(self, left, right, p, bound, threshold=1e-4):
        mid = (left + right) / 2
        mid_kl = self.kld(p, mid)
        limit = bound - mid_kl
        while not ((np.abs(limit) < threshold).all() and (np.abs(left - right) < threshold).all()):
            left, right = np.where((limit > 0), mid, left), np.where((limit > 0), right, mid) 
            mid = (left + right) / 2
            mid_kl = self.kld(p, mid)
            limit = bound - mid_kl
        return mid
    
    def run_algo(self):
        if self.pulls < self.num_arms:
            return self.pulls # sample all the arms exactly once while starting

        pt = self.rewards / self.arm_pulls  

        if (pt == 1).any():
            return self.tie_breaker(pt)
        
        z_idxs = (pt == 0)
        nz_idxs = (pt != 0)
        q_opt = np.zeros_like(pt)
        bound = (np.log(self.pulls) + self.c * np.log(np.log(self.pulls))) / self.arm_pulls
        q_opt[z_idxs] = 1 - np.exp(-bound[z_idxs])
        if (nz_idxs).any():
            q_opt[nz_idxs] = self.bin_search(pt[nz_idxs], np.ones_like(q_opt[nz_idxs]), pt[nz_idxs], bound[nz_idxs])
        return self.tie_breaker(q_opt) 

class Thompson(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
    
    def run_algo(self):
        return self.tie_breaker(np.random.beta(self.rewards + 1, self.arm_pulls - self.rewards + 1))

class ThompsonHint(Algorithms):
    def __init__(self, num_arms, hint):
        super().__init__(num_arms)
        self.hint = hint
        self.p_arg = np.argmax(self.hint)
        # print(self.p_arg)
        self.mypdf = np.zeros((self.num_arms, self.num_arms))
        self.mypdf.fill(1 / self.num_arms)
    
    def run_algo(self):
        if self.pulls == 0:
            return self.tie_breaker(np.random.beta(self.rewards + 1, self.arm_pulls - self.rewards + 1))
        
        if self.rt == 1:
            self.mypdf[self.at] *= self.hint
        else:
            self.mypdf[self.at] *= (1 - self.hint)
        
        self.mypdf[self.at] = self.mypdf[self.at] / np.sum(self.mypdf[self.at])
        pos = self.mypdf[:, self.p_arg]       
        return self.tie_breaker(pos)