import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

HORIZON = [100, 400, 1600, 6400, 25600, 102400]
ALGORITHM_NAMES1 = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
ALGORITHM_NAMES2 = ['thompson-sampling', 'thompson-sampling-with-hint']
IP_NAMES = ['Instance 1', 'Instance 2', 'Instance 3']


algos = ALGORITHM_NAMES1
regs = np.zeros (((3, len(algos), 6)))

data = pd.read_csv('submission/outputDataT1.txt', delimiter=', ', header=None, engine='python')
data.columns = ['f_name', 'algo', 'seed', 'epsilon', 'horizon', 'regret']
df = pd.DataFrame(data)

m = 0
# T4_a
for i in range(3):
    for j in range(len(algos)):
        for k in range(6):
            regs[i, j, k] = df['regret'][m:m+50].mean()
            m += 50

for i in range(3):
    line1, = plt.plot(HORIZON, regs[i][0], 'orange')
    line2, = plt.plot(HORIZON, regs[i][1], 'g')
    line3, = plt.plot(HORIZON, regs[i][2], 'r')
    line4, = plt.plot(HORIZON, regs[i][3], 'b')
    plt.legend((line1, line2, line3, line4), algos)
    plt.xscale('log')
    plt.xlabel('Horizon (T) in log_Scale')
    plt.ylabel('Avg. Regret')
    plt.title(IP_NAMES[i])
    # plt.savefig(IP_NAMES[i]+'_T4_a.png')
    plt.show()


algos = ALGORITHM_NAMES2
regs = np.zeros (((3, len(algos), 6)))
data = pd.read_csv('submission/outputDataT2.txt', delimiter=', ', header=None, engine='python')
data.columns = ['f_name', 'algo', 'seed', 'epsilon', 'horizon', 'regret']
df = pd.DataFrame(data)
m = 0
# T4_b
for i in range(3):
    for j in range(len(algos)):
        for k in range(6):
            regs[i, j, k] = df['regret'][m:m+50].mean()
            m += 50

for i in range(3):
    line1, = plt.plot(HORIZON, regs[i][0], 'r')
    line2, = plt.plot(HORIZON, regs[i][1], 'b')
    plt.legend((line1, line2, line3, line4), algos)
    plt.xscale('log')
    plt.xlabel('Horizon (T) in log_Scale')
    plt.ylabel('Avg. Regret')
    plt.title(IP_NAMES[i])
    # plt.savefig(IP_NAMES[i]+'_T4_b.png')
    plt.show()

# print(regs)
