'''
Author: Manas Vashistha
'''
import numpy as np
from matplotlib import pyplot as plt
from gridworld import GridWorld
from agents import SARSA, ExpSARSA, QLearning

if __name__ == "__main__":
    WINDS = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    GAMMA = 1
    ALPHA = 0.5
    EPS = 0.1
    NUM_ACTIONS = [4, 8, 8]
    MOVES = ['Baseline(4)', 'Kings(8)', 'Kings(8)']
    NUM_EPISODES = 1000
    COLS = 10
    ROWS = 7
    START = (0, 3)
    END = (7, 3)
    
    ST = ['Non', 'Non', '']
    NUM_SEEDS = 10

    AGENT = [[SARSA], [SARSA], [SARSA], [SARSA, ExpSARSA, QLearning]]
    AGENT_NAME = [['SARSA'], ['SARSA'], ['SARSA'], ['SARSA', 'Expected SARSA', 'Q-Learning']]
    STOCHASTIC = [False, False, True, False]
    MOVES = ['Baseline(4)', 'Kings(8)', 'Kings(8)', 'Baseline(4)']
    NUM_ACTIONS = [4, 8, 8, 4]
    ST = ['Non', 'Non', '', 'Non']

    for task in range(4):
        EPISODES = np.zeros((((4, len(AGENT[task]), NUM_SEEDS, NUM_EPISODES))))
        for a in range(len(AGENT[task])):
            for s in range(NUM_SEEDS):
                np.random.seed(s)
                gw = GridWorld(COLS, ROWS, WINDS, NUM_ACTIONS[task], START, END, STOCHASTIC[task])
                agent = AGENT[task][a](ALPHA, gw, GAMMA)
                for e in range(NUM_EPISODES):
                    eps = EPS
                    iteration = agent.execute(eps)
                    if e == 0:
                        EPISODES[task][a][s][e] += iteration
                    else:
                        EPISODES[task][a][s][e] += iteration + EPISODES[task][a][s][e-1]
                print(f"Agent: {AGENT_NAME[task][a]}")
                print(f"Seed: {s}")
                print(f"Total iterations: {iteration}\n")

        for k in range(len(AGENT[task])):
            plt.plot(EPISODES[task][k].mean(0), range(EPISODES[task][k].mean(0).shape[0]), label=AGENT_NAME[task][k])
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.title(f"{MOVES[task]} Moves, {ST[task]} STOCHASTIC")
        plt.grid()
        plt.savefig(f'task{task+2}.png')
        plt.clf()
        # plt.show()