'''
Author: Manas Vashistha
'''
 
import numpy as np
import sys
from mdp import MDP
from algorithms import VI, LP, HPI

if __name__ == '__main__':
    ALGORITHMS = {'vi': VI, 'lp': LP, 'hpi': HPI}

    opts = sys.argv[1::2]
    args = sys.argv[2::2]

    for  i in range(len(opts)):
        opt = opts[i]
        arg = args[i]
        if opt == '--mdp':
            mdp_path = str(arg)
        if opt == '--algorithm':
            algorithm = str(arg)
            assert algorithm in ALGORITHMS, 'not a valid algorithm'


    mdp_instance = MDP(mdp_path)


    PLANNER = ALGORITHMS[algorithm](mdp_instance.T, mdp_instance.R, mdp_instance.gamma, mdp_instance.endstates)

    V_star, pi_star = PLANNER.run()

    for i, j in zip(V_star, pi_star):
        print('{0:.6f}\t{1:0.0f}\n'.format(i, j))