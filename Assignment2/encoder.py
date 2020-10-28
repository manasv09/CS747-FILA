'''
Author: Manas Vashistha
'''

import numpy as np
import sys

if __name__ == '__main__':
    opts = sys.argv[1::2]
    args = sys.argv[2::2]

    for  i in range(len(opts)):
        opt = opts[i]
        arg = args[i]
        if opt == '--grid':
            gridfile = str(arg)
        

    i = 0
    global start
    global end
    global transitions
    global state

    start = -1
    end = []
    transitions = []
    maze = []

    f = open(gridfile, 'r')
    for line in f:
        bits = line.strip().split(' ')
        maze.append([int(b) for b in bits])

    f.close()
    maze = np.array(maze)

    state = {}
    k = 0
    count = 0 
    rew = 100*(maze.shape[0] * maze.shape[1] - maze.sum())
    penalty = -0.5
    wall_penalty = -0.5
    gamma = 0.99
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 1:
                if i > 0:
                    if maze[i-1, j] == 0 or maze[i-1, j] == 2:
                        transitions.append([state[(i-1, j)], 2, state[(i-1, j)], wall_penalty, 1])
                if j > 0:
                    if maze[i, j-1] == 0 or maze[i, j-1] == 2:
                        transitions.append([state[(i, j-1)], 1, state[(i, j-1)], wall_penalty, 1])
                count += 1
                continue
            state[(i, j)] = k
            k += 1
            if maze[i, j] == 2 or maze[i, j] == 0:
                if maze[i, j] == 2:
                    start = state[(i, j)]
                if i > 0:
                    if maze[i-1, j] == 0 or maze[i-1, j] == 2:
                        transitions.append([state[(i, j)], 0, state[(i-1, j)], penalty, 1])
                        transitions.append([state[(i-1, j)], 2, state[(i, j)], penalty, 1])
                    elif maze[i-1, j] == 3:
                        transitions.append([state[(i, j)], 0, state[(i-1, j)], rew, 1])
                    elif maze[i-1, j] == 1:
                        transitions.append([state[(i, j)], 0, state[(i, j)], wall_penalty, 1])
                if j > 0:
                    if maze[i, j-1] == 0 or maze[i, j-1] == 2:
                        transitions.append([state[(i, j)], 3, state[(i, j-1)], penalty, 1])
                        transitions.append([state[(i, j-1)], 1, state[(i, j)], penalty, 1])
                    elif maze[i, j-1] == 3:
                        transitions.append([state[(i, j)], 3, state[(i, j-1)], rew, 1])
                    elif maze[i, j-1] == 1:
                        transitions.append([state[(i, j)], 3, state[(i, j)], wall_penalty, 1])
            elif maze[i, j] == 3:
                end.append(state[(i, j)])
                if i > 0:
                    if maze[i-1, j] == 0 or maze[i-1, j] == 2:
                        transitions.append([state[(i-1, j)], 2, state[(i, j)], rew, 1])
                if j > 0:
                    if maze[i, j-1] == 0 or maze[i, j-1] == 2:
                        transitions.append([state[(i, j-1)], 1, state[(i, j)], rew, 1])


    print('numStates {}'.format(len(state)))
    print('numActions {}'.format(4))
    print('start {}'.format(start))

    for i in end:
        print('end {}'.format(i))

    for transition in transitions:
        print('transition {} {} {} {} {}'.format(transition[0], transition[1], transition[2], transition[3], transition[4]))

    print('mdptype episodic')
    print('gamma  {}'.format(gamma))
