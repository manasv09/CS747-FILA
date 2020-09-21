import numpy as np

class Algorithms:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_pulls = np.zeros(self.num_arms)
        self.pulls = 0
        self.rewards = np.zeros(self.num_arms)

    def execute(self, instance):
        at = self.algo()
        rt = instance.arm_pull(at)
        self.arm_pulls[at] += 1
        self.rewards[at] += rt
        self.pulls = np.sum(self.pulls_count)
    
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
        else:
            return np.argmax(self.rewards / self.arm_pulls)
        
class UCB(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
        self.pt = self.rewards / self.arm_pulls
        self.ucbt = self.pt + np.sqrt(2 * np.log(self.pulls) / self.arm_pulls)
    
    def run_algo(self):
        if self.pulls < self.num_arms:
            return self.pulls # sample all the arms exactly once while starting
        else:
            return self.tie_breaker(self.ucbt) 

class KLUCB(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
    
    def run_algo(self):
        pass

class Thompson(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
    
    def run_algo(self):
        return self.tie_breaker(np.random.beta(self.rewards + 1, self.arm_pulls - self.rewards + 1))

class ThompsonHint(Algorithms):
    def __init__(self, num_arms):
        super().__init__(num_arms)
    
    def run_algo(self):
        pass