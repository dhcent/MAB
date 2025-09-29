import random as rand
import numpy as np
class Greedy:
    def __init__(self, counts = None, states = None, n_arms = 0):
        #counts is # of times each arm was visited
        self.name = "Greedy"
        self.n_arms = n_arms
        self.counts = [0] * self.n_arms if counts is None else counts
        self.states = [0] * self.n_arms if states is None else states

    #reinitialize count and num of arms
    def reinitialize(self, counts = None, means = None):
        self.counts = [0] * self.n_arms if counts is None else counts
        self.states = [0] * self.n_arms if means is None else means
        return

    def select_arm(self):
        if any(count == 0 for count in self.counts):
            return self.counts.index(0)
        max_num = max(self.states)
        return self.states.index(max_num)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.states[arm] = reward
        return

