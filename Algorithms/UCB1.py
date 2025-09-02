#empty for now
import random as rand
import numpy as np
#fixed EpsilonGreedy implementation. Maybe look into Epsilon Decay.
class UCB1:
    def __init__(self, counts = None, means = None, UCB_vals = None, n_arms = 0):
        #counts is # of times each arm was visited
        self.name = "UCB1"
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms) if counts is None else counts
        self.means = np.zeros(n_arms) if means is None else means
        self.UCB_vals = np.zeros(n_arms) if UCB_vals is None else UCB_vals
        self.rounds = 0


    #reinitialize count and num of arms
    def reinitialize(self, counts = None, means = None, UCB_vals = None):
        self.counts = np.zeros(self.n_arms) if counts is None else counts
        self.means = np.zeros(self.n_arms) if means is None else means
        self.rounds = 0
        self.UCB_vals = np.zeros(self.n_arms) if UCB_vals is None else UCB_vals
        return

    def select_arm(self):
        self.rounds += 1
        if self.rounds <= self.n_arms:
            return self.rounds - 1 #account for incrementing before pulling
        return self.UCB_vals.index(max(self.UCB_vals))


    def update(self, arm, reward):
        self.counts[arm]+=1
        n = self.counts[arm]
        #update mean for chosen arm
        new_mean = (self.means[arm] * (n - 1) + reward)/n
        self.means[arm] = new_mean
        #calculate new UCB for ALL arms
        for k in range (self.n_arms):
            if self.counts[k] == 0:
                continue
            self.UCB_vals[k] = self.means[k] + np.sqrt(2 * np.log(self.rounds) / self.counts[k])
        return

