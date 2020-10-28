'''
Author: Manas Vashistha
'''
 
import numpy as np
import pulp

class Algorithms:

    def __init__(self, T, R, gamma, end_states):
        np.random.seed(0)
        self.num_states = T.shape[0]
        self.num_actions = T.shape[1]
        self.T = T
        self.R = R
        self.gamma = gamma
        self.pdt1 = self.T * self.R
        self.pdt2 = self.T * self.gamma
        self.pi = np.random.randint(self.num_actions, size=self.num_states, dtype=np.int32)
        if end_states == -1:
            self.end_states = []
        else:
            self.end_states = end_states
    
    def run(self):
        return self.solve()

    def solve(self):
        raise NotImplementedError

    def Q_pi(self, a, V):
        if (a == np.arange(self.num_actions)).all():
            return self.pdt1.sum(2) + (self.pdt2 * V[np.newaxis, np.newaxis, :]).sum(2)
        return (self.T[:, a, :] * (self.R[:, a, :]).sum(2) + (self.gamma * V[np.newaxis, np.newaxis, :])).sum(2)

    def get_policy(self, V):
        pi_star = np.argmax(self.Q_pi(np.arange(self.num_actions), V), axis=1)
        return pi_star


class VI(Algorithms):

    def __init__(self, T, R, gamma, end_states):
        super().__init__(T, R, gamma, end_states)
        
    def solve(self):
        if self.gamma != 1 and self.gamma != 0:
            dtype = np.float32
            A = self.pdt1.astype(dtype).sum(2)
            B = self.pdt2.astype(dtype)
            bound = 1e-6*(1/self.gamma - 1)
        else:
            A = self.pdt1.sum(2)
            B = self.pdt2
            bound = 1e-8
        V = A.max(1)
        while True:
            V_updated = (A + (B * V[np.newaxis, np.newaxis, :]).sum(2)).max(1)
            if (abs(V_updated - V)).max() < bound:
                break
            else:
                V = V_updated
        return V_updated, self.get_policy(V_updated)


class LP(Algorithms):

    def __init__(self, T, R, gamma, end_states):
        super().__init__(T, R, gamma, end_states)
        
    def solve(self):
        prob = pulp.LpProblem('MDPPlanning', pulp.LpMaximize)
        variable = pulp.LpVariable.dict('v', np.arange(self.num_states))
        prob += pulp.lpSum([-v for v in variable.values()])
        var = np.array(list(prob.variables()))
        arr = self.pdt1.sum(2) + (self.pdt2 * var).sum(2)
        for i in range(var.shape[0]):
            for j in range(self.num_actions):
                prob += var[i] - arr[i, j] >= 0
        pulp.PULP_CBC_CMD(msg=0).solve(prob)
        V_star = np.zeros(self.num_states)
        for i, variable in enumerate(prob.variables()):
            V_star[i] = variable.varValue
        return V_star, self.get_policy(V_star)


class HPI(Algorithms):

    def __init__(self, T, R, gamma, end_states):
        super().__init__(T, R, gamma, end_states)
    
    def bellman_solver(self):
        idxs = list(range(self.num_states))
        for i in self.end_states:
            if i in idxs:
                idxs.remove(i)
        idxs = np.array(idxs)
        x = np.zeros(self.num_states)
        eye = np.zeros(((len(idxs), len(idxs))))
        eye = np.identity(len(idxs))

        A = eye - self.pdt2[idxs, self.pi[idxs]][:, idxs]
        B = self.pdt1[idxs, self.pi[idxs]].sum(1)
        try:
            x[idxs] = np.linalg.solve(A, B)
        except:
            x[idxs] = np.linalg.lstsq(A, B, rcond=-1)[0]
        return x
    
    def solve(self):
        np.random.seed(1)
        new_pi = np.random.randint(self.num_actions, size=self.num_states, dtype=np.int32)
        while (new_pi != self.pi).any():
            self.pi = new_pi
            V = self.bellman_solver()
            new_pi = self.get_policy(V)
        return V, self.pi