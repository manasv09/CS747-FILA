'''
Author: Manas Vashistha
'''
import numpy as np

class MultiArmedBandit:
    def __init__(self, instance_path):
        ''' 
        Initialise MultiArmedBandit with params:

        instance_path: PATH OF THE FILE WHERE THE BANDIT RESIDES.
        '''
        f = open(instance_path, 'r')
        self.means = np.array([float(p.strip()) for p in f.readlines()]) # read the means from the file
        self.num_arms = len(self.means) # number of arms of the bandit
        self.p_star = np.max(self.means) # the optimal mean
        f.close()

    def pull_arm(self, armi):
        '''
        Pulls a single arm of the bandit whose index is given by armi.

        armi: Arm pulled

        return 1 if he the value sampled from random [0, 1] is less than the mean of armi else 0.
        '''
        if armi < self.num_arms:
            if np.random.uniform(0, 1) < self.means[armi]:
                return 1
            else:
                return 0