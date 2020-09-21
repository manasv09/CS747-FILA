import numpy as np
import getopt
import sys
from bandit_instance import MultiArmedBandit
from algorithms import *

ALGORITHMS = ['epsilon-greedy', 'ucb', 'kl-ucb', ' thompson-sampling', ' thompson-sampling-with-hint']

opts = sys.argv[1::2]
args = sys.argv[2::2]

for  i in range(len(opts)):
    opt = opts[i]
    arg = args[i]
    if opt == '--instance':
        instance_path = str(arg)
        print('\ninstance:', instance_path)
    if opt == '--algorithm':
        algorithm = str(arg)
        assert algorithm in ALGORITHMS, 'randomSeed must be a non-negative integer'
        print('algorithm:', algorithm)
    if opt == '--randomSeed':
        randomSeed = int(arg)
        assert randomSeed >= 0, 'randomSeed must be a non-negative integer'
        print('randomSeed:', randomSeed)
    if opt == '--epsilon':
        epsilon = float(arg)
        assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in the range [0, 1]'
        print('epsilon:', epsilon)
    if opt == '--horizon':
        horizon = float(arg)
        assert horizon >= 0, 'horizon must be a non-negative integer'
        print('horizon:', horizon)

np.random.seed(randomSeed)    

b = MultiArmedBandit(instance_path)
print(b.pull_arm(0))

