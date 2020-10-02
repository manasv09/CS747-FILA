'''
Author: Manas Vashistha
'''
import numpy as np
import sys
# import time
from bandit_instance import MultiArmedBandit
from algorithms import EpsilonGreedy, UCB, KLUCB, Thompson, ThompsonHint

# a = time.time()

# The algorithms
ALGORITHMS = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling', 'thompson-sampling-with-hint']

# Input args
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

# Seeded the np random with given seed
np.random.seed(randomSeed)   
# created an instance of MultiArmedBandit to be run for horizon pulls and pass the input file path
b = MultiArmedBandit(instance_path)

# decided which algo to run
if algorithm == 'epsilon-greedy':
    alg = EpsilonGreedy(epsilon, b.num_arms)
elif algorithm == 'ucb':
    alg = UCB(b.num_arms)
elif algorithm == 'kl-ucb':
    alg = KLUCB(b.num_arms)
elif algorithm == 'thompson-sampling':
    alg = Thompson(b.num_arms)
elif algorithm == 'thompson-sampling-with-hint':
    hint = np.sort(b.means) # created a hint array for passing to Thompson hint
    alg = ThompsonHint(b.num_arms, hint)


for t in range(horizon):
    alg.execute(b) # executed the bandit instance for each pull

regret = horizon*b.p_star - alg.cum_rew # calculted regret

print('{}, {}, {}, {}, {}, {}\n'.format(instance_path, algorithm, randomSeed, epsilon, horizon, regret)) # print op
# print(time.time() - a)
