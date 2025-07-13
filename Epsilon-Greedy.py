import random as rand

class EpsilonGreedy:
    def __init__(self, epsilon, counts = None, means = None, n_arms = 0):
        #counts is # of times each arm was visited
        self.epsilon = epsilon
        self.counts = counts
        self.means = means
        self.n_arms = n_arms
        return

    #reinitialize count and num of arms
    def initialize(self, counts = None, means = None, n_arms=0):
        self.n_arms = n_arms
        if counts is None:
            self.counts = [0] * self.n_arms
        else:
            self.counts = counts
        if means is None:
            self.means = [0] * self.n_arms
        else:
            self.means = means
        return

    def select_arm(self):
        if rand.random() > self.epsilon:
            max_num = max[self.means]
            return self.means.index(max_num)
        else:
            rand.randint(0,self.n_arms)
            return

    def update(self, arm, reward):
        self.counts[arm]+=1
        n = self.counts[arm]
        #update mean
        new_mean = (self.means[arm] * (n - 1) + reward)/n
        self.means[arm] = new_mean
        return



class BernoulliArm:
    def __init__(self, p):
        self.p = p

    # Reward system based on Bernoulli (copied from:
    # https://github.com/kfoofw/bandit_simulations/blob/master/python/multiarmed_bandits/analysis/eps-greedy.md)
    def draw(self):
        if rand.random() > self.p:
            return 0
        else:
            return 1

#horizon is # of steps or rounds that alg will run
def run_algorithm(arms, algo, horizon):
    total_reward = 0
    chosen_arms = []
    cumulative_rewards = []

    algo.initialize(len(arms))

    for t in (horizon):
        chosen_arm = algo.select_arm()