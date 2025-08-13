#algorithm imports
from Algorithms import *

#arm imports
import Arms

#plot imports
import Plots

# from Arms.bernoulli import BernoulliArm
import numpy as np
import random as rand
import matplotlib.pyplot as plt


# horizon is total # of steps
def run_algorithm(algorithm, horizon, arms, epoch):
    total_reward = 0
    oracle_total_reward = 0
    oracle = Oracle(arms)
    # create a matrix of nxm
    # n - # of arms
    # m - # of steps (measures each mean at each step, used for plotting later)
    mean_history = [[] for i in range(len(arms))]
    regret_history = []
    cumulative_regret_history = []

    for t in range (horizon):
        # ONLY FOR BROWNIAN ARMS
        if t != 0 and t % epoch == 0:
            # print(f"Algorithm: {algorithm.select_arm()}")
            # print(f"Oracle: {oracle.select_arm()}")
            algorithm.reinitialize(n_arms=len(arms))
           # print("Reinitialized!")

        for arm in arms:
            arm.simulate_brownian()
        #chosen arm is index
        chosen_arm_index = algorithm.select_arm()
        reward = arms[chosen_arm_index].draw()
        total_reward += reward

        #oracle action
        oracle_chosen_arm_index = oracle.select_arm()
        algorithm.update(chosen_arm_index, reward)

        regret = arms[oracle_chosen_arm_index].mean - arms[chosen_arm_index].mean
        #used for plotting regret
        if t == 0:
            cumulative_regret_history.append(regret)
        else:
            cumulative_regret_history.append(regret + cumulative_regret_history[t - 1])
        regret_history.append(regret)

        #used for plotting arm means
        for i in range(len(arms)):
            mean_history[i].append(arms[i].mean)

    #plot regret vs log rounds
    Plots.plot_regret(cumulative_regret_history, "cumulative_regret")
    print(cumulative_regret_history)
    Plots.plot_regret(regret_history)
    Plots.plot_arm_means(mean_history)

#horizon is # of steps or rounds that alg will run
arms = []
#n - num of arms
n = 5
#generate each arm based on Arm Type and n # of arms

expected_vals = [rand.random() for i in range(n)]
std_dev = 0.1
drift = 0.01
volatility = 0.5
for val in expected_vals:
    arms.append(Arms.BrownianArm(val, std_dev, drift, volatility))


#given algorithm + horizon
eps = 0.05
horizon = 3000
#loop through # of arms

#calculate epoch length DOESNT MAKE SENSE
avg_vol = 0
for arm in arms:
    avg_vol += arm.volatility ** 2
avg_vol *= 1/n
avg_vol = np.sqrt(avg_vol)

epoch_length = avg_vol * np.sqrt(np.log(1 / avg_vol))
epoch_length = 500

#Algorithms
eps_algo = EpsilonGreedy(eps, n_arms = len(arms))
UCB_algo = UCB1(n_arms = len(arms))


run_algorithm(eps_algo, horizon, arms, epoch_length)
#print(epoch_length)
