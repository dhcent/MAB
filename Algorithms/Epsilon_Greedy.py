import random as rand
import numpy as np
#fixed EpsilonGreedy implementation. Maybe look into Epsilon Decay.
class EpsilonGreedy:
    def __init__(self, epsilon, counts = None, means = None, n_arms = 0):
        #counts is # of times each arm was visited
        self.name = "EpsilonGreedy"
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = [0] * self.n_arms if counts is None else counts
        self.means = [0] * self.n_arms if means is None else means

    #reinitialize count and num of arms
    def reinitialize(self, counts = None, means = None):
        self.counts = [0] * self.n_arms if counts is None else counts
        self.means = [0] * self.n_arms if means is None else means
        return

    def select_arm(self):
        #force algorithm to pull arms once each time
        if any(count == 0 for count in self.counts):
            return self.counts.index(0)
        if rand.random() > self.epsilon:
            max_num = max(self.means)
            return self.means.index(max_num)
        else:
            #pick random arm
            return rand.randint(0,self.n_arms - 1)

    def update(self, arm, reward):
        self.counts[arm]+=1
        n = self.counts[arm]
        #update mean
        new_mean = (self.means[arm] * (n - 1) + reward)/n
        self.means[arm] = new_mean
        return

