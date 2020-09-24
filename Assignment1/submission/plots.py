import numpy as np
import getopt
import time
import sys
from bandit_instance import MultiArmedBandit
from algorithms import EpsilonGreedy, UCB, KLUCB, Thompson, ThompsonHint
from matplotlib import pyplot as plt

h = [100, 400, 1600, 6400, 25600, 102400]

algs1 = [EpsilonGreedy, UCB, KLUCB, Thompson]
ALGORITHMS1 = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling', 'thompson-sampling-with-hint']

algs2 = [Thompson, ThompsonHint]
ALGORITHMS2 = ['thompson-sampling', 'thompson-sampling-with-hint']


files = ['../instances/i-1.txt', '../instances/i-2.txt', '../instances/i-3.txt']
op_file = ['outputDataT1.txt', 'outputDataT2.txt']
n = [2, 5, 25]
eps = 0.6

algs = algs1
ALGORITHMS = ALGORITHMS1

seed = np.arange(50)
r_avg = np.zeros((len(algs), len(h)))
regrets = np.zeros((len(algs), len(h)))

opf = open(op_file[0], 'w')
for m, f in enumerate(files):
    # e = 0
    for j, algo in enumerate(algs):
        for i, hi in enumerate(h):
            a = time.time()
            r = 0
            reg = 0
            opti = np.zeros(n[m])
            for s in seed:
                np.random.seed(s)
                b = MultiArmedBandit(f)
                if algo == ThompsonHint:
                    hint = np.sort(b.means)
                    alg = algo(b.num_arms, hint)
                elif algo == EpsilonGreedy:
                    alg = algo(eps, b.num_arms)
                else:
                    alg = algo(b.num_arms)
                for it in range(hi):
                    alg.execute(b)
                # print(alg.arm_pulls)
                msg = f + ', ' + ALGORITHMS[j] + ', ' + str(s) + ', ' + str(eps) + ', ' + str(hi) + ', ' + str(hi*b.p_star - alg.cum_rew) + '\n'
                opf.write(msg)
                r += alg.cum_rew
                reg += hi*b.p_star - alg.cum_rew
                opti += alg.arm_pulls
                # print(opti)
            print('\n{}/{}'.format(opti, len(seed)*hi))
            r_avg[j][i] = r / len(seed)
            regrets[j][i] = reg / len(seed)
            print('Reward for horizon {} = {}'.format(hi, r_avg[j][i]))
            print('Regret for horizon {} = {}'.format(hi, regrets[j][i]))
            print('Time taken: {}'.format(time.time()-a))
            
        # e += 1
    
    # for k in range(len(algs)):
    #     plt.plot(h, r_avg[k])
    # plt.grid()
    # plt.savefig('rewards' + str(m) + '.png')
    # plt.show()
opf.close()
    # for k in range(len(algs)):
    #     plt.plot(h, regrets[k])
    # plt.grid()
    # plt.savefig('regrets' + str(m) + '.png')
    # plt.show()

# for i in range(len(algs)):
#     plt.plot(np.log(h), (r_avg[i]))
# plt.grid()
# # plt.savefig('log_rewards3.png')
# plt.show()

# for i in range(len(algs)):
#     plt.plot(np.log(h), (regrets[i]))
# plt.grid()
# # plt.savefig('log_regrets3.png')
# plt.show()
