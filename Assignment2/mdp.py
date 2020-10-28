'''
Author: Manas Vashistha
'''
 
import numpy as np

class MDP:

    def __init__(self, mdp_path):
        f = open(mdp_path, 'r')
        self.read(f)
        f.close()
        
    def read(self, f):
        self.num_states = int(f.readline().strip().split(' ')[-1])
        self.num_actions = int(f.readline().strip().split(' ')[-1])
        self.start_state = int(f.readline().strip().split(' ')[-1])
        self.endstates = [int(k) for k in f.readline().strip().split(' ')[1:]]

        transitions = []
        line = f.readline().strip().split(' ')
        while line[0] != 'mdptype':
            transitions.append([int(line[1]), int(line[2]),int(line[3]), float(line[4]), float(line[5])])
            line = f.readline().strip().split(' ')
        transitions = np.array(transitions)

        self.mdptype = str(line[1])
        self.gamma = float(str(f.readline().strip().split(' ')[2]))

        self.T = np.zeros(((self.num_states, self.num_actions, self.num_states)))
        self.R = np.zeros(((self.num_states, self.num_actions, self.num_states)))
        for transition in transitions:
            self.T[int(transition[0]), int(transition[1]), int(transition[2])] = transition[4]
            self.R[int(transition[0]), int(transition[1]), int(transition[2])] = transition[3]