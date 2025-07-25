from Algorithms.Epsilon_Greedy import EpsilonGreedy
from Arms.normal import NormalArm
from Algorithms.Oracle import Oracle
from Arms.gaussian import GaussianArm
from Arms.brownian import BrownianArm

# from Arms.bernoulli import BernoulliArm
import numpy as np
import random as rand
import matplotlib.pyplot as plt




def run_algorithm(algorithm, horizon):
    total_reward = 0
    oracle_total_reward = 0
    cumulative_rewards = [0] * len(arms)
    oracle = Oracle(arms)
    algorithm.reinitialize(n_arms=len(arms))
    #horizon is total # of steps
    for t in range (horizon):
        #chosen arm is index
        chosen_arm = algorithm.select_arm()
        reward = arms[chosen_arm].draw()
        total_reward += reward

        #oracle action
        oracle_chosen_arm = oracle.select_arm()
        oracle_reward = arms[oracle_chosen_arm].draw()
        oracle_total_reward += oracle_reward
        algorithm.update(chosen_arm, reward)

        #plotting
        plt.ylabel("regret")
        plt.xlabel("t rounds")
        # plt.plot(t, total_reward, 'bo')
        # plt.plot(t, oracle_total_reward, "ro")
        #plotting regret
        regret = oracle_total_reward - total_reward
        plt.plot(t, regret, "go")
        cumulative_rewards[chosen_arm] += reward
    print(oracle.select_arm())
    plt.ylim(0, regret)
    plt.xlim(0, regret)

    plt.show()
    return cumulative_rewards
#horizon is # of steps or rounds that alg will run
arms = []
expected_vals = [rand.random() * 5 for i in range(30)]
for val in expected_vals:
    arms.append(BrownianArm(val))

    #given algorithm + horizon
eps = 0.1
horizon = 5000
algo = EpsilonGreedy(eps, n_arms = len(arms))
run_algorithm(algo, horizon)

for i in range (len(arms)):
    print(f"{i}: {arms[i].mean}")
#print(max(arms))