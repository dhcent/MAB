#algorithm imports
import Algorithms as algo

#arm imports
import Arms

#plot imports
from Plots.regret import plot_regret
from Plots.regret import plot_regret_log_t
from Plots.arm_means import plot_arm_means

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

    for t in range (horizon):
        # ONLY FOR BROWNIAN ARMS
        if t != 0 and t % epoch == 0:
                algorithm.reinitialize(n_arms=len(arms))
        for arm in arms:
            arm.simulate_brownian()
        #chosen arm is index
        chosen_arm = algorithm.select_arm()
        reward = arms[chosen_arm].draw()
        total_reward += reward

        #oracle action
        oracle_chosen_arm = oracle.select_arm()
        oracle_reward = arms[oracle_chosen_arm].draw()
        oracle_total_reward += oracle_reward
        algorithm.update(chosen_arm, reward)

        #used for plotting regret
        if t == 0:
            regret_history.append(oracle_reward - reward)
        else:
            regret_history.append(oracle_reward - reward + regret_history[t - 1])

        #used for plotting arm means
        for i in range(len(arms)):
            mean_history[i].append(arms[i].mean)
        print(algorithm.select_arm())
        print(oracle.select_arm())
    #plot regret vs log rounds
    plot_regret_log_t(regret_history)
    plot_arm_means(mean_history)

#horizon is # of steps or rounds that alg will run
arms = []
#n - num of arms
n = 5
#generate each arm based on Arm Type and n # of arms

expected_vals = [rand.random() for i in range(n)]
std_dev = 0.1
drift = 0
volatility = 0.01
for val in expected_vals:
    arms.append(Arms.BrownianArm(val, std_dev, drift, volatility))

    #given algorithm + horizon
eps = 0.01
horizon = 5000
#loop through # of arms

#calculate epoch length
avg_vol = 0
for arm in arms:
    avg_vol += arm.volatility ** 2
avg_vol *= 1/n
avg_vol = np.sqrt(avg_vol)

epoch_length = avg_vol * np.sqrt(np.log(1 / avg_vol))
epoch_length = 50

#Algorithms
eps_algo = algo.EpsilonGreedy(eps, n_arms = len(arms))
UCB_algo = algo.UCB1(n_arms = len(arms))


run_algorithm(eps_algo, horizon, arms, epoch_length)
print(epoch_length)
