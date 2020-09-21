import numpy as np

class MultiArmedBandit:
    def __init__(self, instance_path):
        f = open(instance_path, 'r')
        self.means = np.array([float(p.strip()) for p in f.readlines()])
        self.num_arms = len(self.means)
        self.p_star = np.max(self.means)
        f.close()
        # print(self.means)
        # print(self.num_arms)
        # print(self.p_star)

    def pull_arm(self, armi):
        if armi < self.num_arms:
            return np.random.binomial(1, self.means[armi])