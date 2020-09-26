import numpy as np
import time
from submission.bandit_instance import MultiArmedBandit
from submission.algorithms import EpsilonGreedy, UCB, KLUCB, Thompson, ThompsonHint

seed = np.arange(50)

HORIZON = [100, 400, 1600, 6400, 25600, 102400]
HORIZON2 = [102400]

eps1 = [0.02]
ALGORITHMS1 = [EpsilonGreedy, UCB, KLUCB, Thompson]
ALGORITHM_NAMES1 = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']

ALGORITHMS2 = [EpsilonGreedy, EpsilonGreedy, EpsilonGreedy]
ALGORITHM_NAMES2 = ['epsilon-greedy', 'epsilon-greedy', 'epsilon-greedy']
eps2 = [0.003, 0.04, 0.6]

ALGORITHMS3 = [Thompson, ThompsonHint]
ALGORITHM_NAMES3 = ['thompson-sampling', 'thompson-sampling-with-hint']

IN_FILES = ['instances/i-1.txt', 'instances/i-2.txt', 'instances/i-3.txt']
OUT_FILES = ['submission/outputDataT1.txt', 'submission/outputDataT2.txt']
IP_NAMES = ['Instance 1', 'Instance 2', 'Instance 3']

exp1 = OUT_FILES[0]
exp2 = OUT_FILES[1]

eps = eps1
h = HORIZON
algos = ALGORITHMS1
algo_names = ALGORITHM_NAMES1
ip_names = IP_NAMES
# opf = open(exp1, 'w')

for i, ipf in enumerate(IN_FILES):
    for j, algo in enumerate(algos):
        e = 0
        for k, hi in enumerate(h):
            rew = 0
            reg = 0
            a = time.time()
            for s in seed:
                np.random.seed(s)
                b = MultiArmedBandit(ipf)
                if algo == EpsilonGreedy:
                    algo_inst = algo(eps[e % len(eps)], b.num_arms)
                elif algo == ThompsonHint:
                    hint = np.sort(b.means)
                    algo_inst = algo(b.num_arms, hint)
                else:
                    algo_inst = algo(b.num_arms)
                for t in range(hi):
                    algo_inst.execute(b)
                r1 = algo_inst.cum_rew
                r2 = hi*b.p_star - r1
                rew += r1
                reg += r2
                msg = ipf+', '+algo_names[j]+', '+str(s)+', '+str(eps[e % len(eps)])+', '+str(hi)+', '+str(r2)+'\n'
                # opf.write(msg)
                # print('\n{}'.format(msg))
            rew_avg = rew / len(seed)
            reg_avg = reg / len(seed)
            print('\nALGORITHM : {}'.format(algo_names[j]))
            print('INPUT : {}'.format(ip_names[i]))
            print('Average Reward for Horizon : {}'.format(rew_avg))
            print('Average Regret for Horizon : {}'.format(reg_avg))
            print('Time taken to execute : {}'.format(time.time() - a))
        e += 1
                

# opf.close()