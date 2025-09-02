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


def generate_arms(means):
    arms = []
    n = 9
    # generate each arm based on Arm Type and n # of arms
    std_dev = 0.1
    drift = 0.01
    volatility = 0.05
    #9 Brownian, 1 Gaussian
    for val in means:
        arms.append(Arms.BrownianArm(val, std_dev, drift, volatility))

    arms.append(Arms.GaussianArm(0.5, 0.1))
    return arms

# horizon is total # of steps
def run_algorithm(algorithm, horizon, arms, epoch = 0):
    total_reward = 0
    oracle_total_reward = 0
    oracle = Oracle(arms)
    # create a matrix of nxm
    # n - # of arms
    # m - # of steps (measures each mean at each step, used for plotting later)

    pulled_rewards = [0 for i in range(len(arms))]
    mean_history = [[] for i in range(len(arms))]
    regret_history = []
    cumulative_regret_history = [0]
    for t in range (horizon):
        if t != 0 and epoch != 0 and t % epoch == 0: #epoch system
            # print(f"Algorithm: {algorithm.select_arm()}")
            # print(f"Oracle: {oracle.select_arm()}")
            algorithm.reinitialize()
           # print("Reinitialized!")

        #draw from each arm
        for i in range(len(arms)):
            if arms[i].name == "BrownianArm":
                arms[i].simulate_brownian()
            pulled_rewards[i] = arms[i].draw()

        #chosen arm is index
        #!!!!!!!!!!!!!! FOR EGREEDY THE BOUND IS STATE-INFORMED
        chosen_arm_index = algorithm.select_arm()
        reward = pulled_rewards[chosen_arm_index]
        algorithm.update(chosen_arm_index, reward)

        total_reward += reward

        #oracle action
        oracle_chosen_arm_index = oracle.select_arm(pulled_rewards)
        oracle_reward = pulled_rewards[oracle_chosen_arm_index]

        # used for plotting regret
        regret = oracle_reward - reward
        if t == 0:
            cumulative_regret_history.append(regret)
        else:
            cumulative_regret_history.append(regret + cumulative_regret_history[t - 1])
        #for steady-state regret
        regret_history.append(regret)

        #used for plotting arm means
        for i in range(len(arms)):
            mean_history[i].append(arms[i].mean)

    #plot regret vs log rounds
    Plots.plot_regret(cumulative_regret_history, algorithm.name, "cumulative_regret")
    Plots.plot_regret(regret_history, algorithm.name)
    Plots.plot_arm_means(mean_history, algorithm.name)

#horizon is # of steps or rounds that alg will run


#computing average based on epoch length
#volatility_avg = np.average(volatilities)
#epoch_length = int(np.ceil(max(n+1, volatility_avg * np.sqrt(np.log(1 / volatility_avg))))) #cast as int and round up**
n = 4
expected_vals = [rand.random() for i in range(n)]
arms1 = generate_arms(expected_vals)
arms2 = generate_arms(expected_vals)
for i in range(len(arms1)):
    print(arms1[i].mean)

#given algorithm + horizon
eps = 0.05
horizon = 1000


eps_algo = EpsilonGreedy(eps, n_arms = len(arms1))
# UCB_algo = UCB1(n_arms = len(arms1))
#EXP3(gamma, n_arms)
EXP3_algo = EXP3(0.1, len(arms1))
epoch = 100
run_algorithm(EXP3_algo, horizon, arms1, epoch)
run_algorithm(eps_algo, horizon, arms2, epoch)
# run_algorithm(UCB_algo, horizon, arms2, epoch)


