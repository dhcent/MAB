import random as rand
import numpy as np
import math

class EXP3:
    def __init__(self, gamma, n_arms = 0, weights = None):
        #counts is # of times each arm was visited
        self.probabilityDistribution = None
        self.name = "EXP3"
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(self.n_arms) if weights is None else np.array(weights)
        self.rounds = 0
        #EXP3 probability distribution calculation
        self.update_probabilities()

    # #reinitialize count and num of arms
    def reinitialize(self, weights = None, gamma = None):
        gamma = self.gamma if gamma is None else gamma
        self.rounds = 0
        self.weights = [1] * self.n_arms if weights is None else weights
        return

    def select_arm(self):
        #Sample an arm according to EXP3
        prob_distr = self.probabilityDistribution
        return np.random.choice(self.n_arms, p=prob_distr)

    #note, arm index starts from 0
    def update(self, arm, reward):
        self.rounds += 1
        #update weights
        prob_distr = self.probabilityDistribution
        estimated_reward = reward / prob_distr[arm] 
        self.weights[arm] *= math.exp(self.gamma * estimated_reward / self.n_arms)
        self.update_probabilities()
        return

    #EXP3 probability calculation
    def update_probabilities(self):
        self.probabilityDistribution = [(1 - self.gamma) * (w / np.sum(self.weights)) + self.gamma / self.n_arms for w
                                        in self.weights]
