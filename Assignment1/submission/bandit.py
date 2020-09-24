import numpy as np
import sys
# import time
from algorithms import EpsilonGreedy, UCB, KLUCB, Thompson, ThompsonHint

class MultiArmedBandit:
    def __init__(self, instance_path):
        f = open(instance_path, 'r')
        self.means = np.array([float(p.strip()) for p in f.readlines()])
        self.num_arms = len(self.means)
        self.p_star = np.max(self.means)
        f.close()

    def pull_arm(self, armi):
        if armi < self.num_arms:
            return np.random.binomial(1, self.means[armi])

# a = time.time()

ALGORITHMS = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling', 'thompson-sampling-with-hint']

opts = sys.argv[1::2]
args = sys.argv[2::2]

for  i in range(len(opts)):
    opt = opts[i]
    arg = args[i]
    if opt == '--instance':
        instance_path = str(arg)
        # print('\ninstance: {}'.format(instance_path))
    if opt == '--algorithm':
        algorithm = str(arg)
        assert algorithm in ALGORITHMS, 'not a valid algorithm'
        # print('algorithm: {}'.format(algorithm))
    if opt == '--randomSeed':
        randomSeed = int(arg)
        assert randomSeed >= 0, 'randomSeed must be a non-negative integer'
        # print('randomSeed: {}'.format(randomSeed))
    if opt == '--epsilon':
        epsilon = float(arg)
        # assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in the range [0, 1]'
        # print('epsilon: {}'.format(epsilon))
    if opt == '--horizon':
        horizon = int(arg)
        assert horizon >= 0, 'horizon must be a non-negative integer'
        # print('horizon: {}\n'.format(horizon))

np.random.seed(randomSeed)    
b = MultiArmedBandit(instance_path)

if algorithm == 'epsilon-greedy':
    alg = EpsilonGreedy(epsilon, b.num_arms)
elif algorithm == 'ucb':
    alg = UCB(b.num_arms)
elif algorithm == 'kl-ucb':
    alg = KLUCB(b.num_arms)
elif algorithm == 'thompson-sampling':
    alg = Thompson(b.num_arms)
elif algorithm == 'thompson-sampling-with-hint':
    hint = np.sort(b.means)
    alg = ThompsonHint(b.num_arms, hint)


for t in range(horizon):
    alg.execute(b)

regret = horizon*b.p_star - alg.cum_rew

print('{}, {}, {}, {}, {}, {}\n'.format(instance_path, algorithm, randomSeed, epsilon, horizon, regret))
# print(time.time() - a)
