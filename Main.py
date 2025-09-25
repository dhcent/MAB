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
def run_algorithm(algorithm, horizon, arms, epoch = 0):
    total_reward = 0
    oracle_total_reward = 0
    oracle = Oracle(arms)

    pulled_rewards = np.zeros(len(arms))
    mean_history = [[] for i in range(len(arms))]
    regret_history = [] #pseudo-regret
    cumulative_regret_history = [0]

    for t in range (horizon):
        if t != 0 and epoch != 0 and t % epoch == 0: #epoch system
            # print(f"Algorithm: {algorithm.select_arm()}")
            # print(f"Oracle: {oracle.select_arm()}")
            algorithm.reinitialize()
           # print("Reinitialized!")

        #simulate arms
        arm_means = np.zeros(len(arms))
        for i in range(len(arms)):
            # simulate brownian
            if arms[i].name == "BrownianArm":
                arms[i].simulate_brownian()
            pulled_rewards[i] = arms[i].draw()
            arm_means[i] = arms[i].mean

        #!!!!!!!!!!!!!! FOR EGREEDY THE BOUND IS STATE-INFORMED
        chosen_arm_index = algorithm.select_arm()

        #STATE INFORMED CASE:
        algorithm.update(chosen_arm_index, arms[chosen_arm_index].mean)

        #STATE OBLIVIOUS CASE:
        #reward = pulled_rewards[chosen_arm_index]
        #algorithm.update(chosen_arm_index, reward)

        #oracle action for pseudo-regret
        oracle_chosen_arm_index = oracle.select_arm(arm_means)

        # note: this is pseudo-regret
        regret = arms[oracle_chosen_arm_index].mean - arms[chosen_arm_index].mean
        #for steady-state regret
        regret_history.append(regret)

        #used for plotting arm means
        for i in range(len(arms)):
            mean_history[i].append(arms[i].mean)
        cumulative_regret_history = np.cumsum(regret_history)

    return regret_history, cumulative_regret_history, mean_history
#horizon is # of steps or rounds that alg will run


def generate_arms(n, type="Gaussian"):
    means = [rand.random() for i in range(n)]
    arms = []
    # generate each arm based on Arm Type and n # of arms
    std_dev = 0.05
    drift = 0
    volatility = 0.75
    #9 Brownian, 1 Gaussian
    if type == "Brownian":
        for i in range(len(means)):
            arms.append(Arms.BrownianArm(means[i], std_dev, drift, volatility))
    elif type == "Gaussian":
        for i in range(len(means)):
            arms.append(Arms.GaussianArm(means[i], std_dev))
    # arms.append(Arms.GaussianArm(means[-1], std_dev, drift))
    return arms, means

n = 5
arm_type = "Brownian"

#NOTE: GENERATED ARMS ARE DIFFERENT WITH THIS CURRENT IMPLEMENTATION
# arms1 = generate_arms(arm_type)
arms2, expected_vals = generate_arms(n, arm_type)

for i in range(len(arms2)):
    print(arms2[i].mean)

#given algorithm + horizon
eps = 0.03
horizon = 8000
epoch = 10000

# eps_algo = EpsilonGreedy(eps, n_arms = len(arms1))
UCB_algo = UCB1(n_arms = len(arms2))
#EXP3(gamma, n_arms)
#EXP3_algo = EXP3(0.1, len(arms1))
# regret_history_1, cumulative_regret_history_1, mean_history_1 = run_algorithm(EXP3_algo, horizon, arms1, epoch)
regret_history_2, cumulative_regret_history_2, mean_history_2 = run_algorithm(UCB_algo, horizon, arms2, epoch)
# run_algorithm(UCB_algo, horizon, arms2, epoch)

# plot cumulative regret history
plt.figure() #makes a clean canvas
#Plots.plot_regret(cumulative_regret_history_1, "EXP3", "Cumulative Regret", color="red")
Plots.plot_regret(cumulative_regret_history_2, "UCB1", y_label="Cumulative Regret", color="blue")

#n = horizon
def calc_UCB_theoretical_regret_bound(n, exp_vals):
    #Theoretical UCB regret bounds
    theoretical_UCB_regret_bounds = 0
    max_val = max(exp_vals)
    for val in exp_vals:
        if val != max_val:
            theoretical_UCB_regret_bounds += np.log(n) / (max_val - val)
    theoretical_UCB_regret_bounds *= 8
    sum_of_difference = 0
    for val in exp_vals:
        sum_of_difference += max_val - val
    theoretical_UCB_regret_bounds += (1 + (np.square(np.pi))/3) * sum_of_difference
    return theoretical_UCB_regret_bounds

UCB_bound = calc_UCB_theoretical_regret_bound(horizon, expected_vals)
print(f"Theoretical UCB Regret Upperbound: {round(UCB_bound, 5)}")
print(f"Actual Cumulative Regret: {round(cumulative_regret_history_2[-1], 5)}")

# For per round regret
# plt.figure()
# Plots.plot_regret(regret_history_1, "EXP3", "Per-round-regret", color="green")
# Plots.plot_regret(regret_history_2, "Epsilon-Greedy", "Per-round-regret", color="pink")
# print(f"AVERAGE REGRET FOR 1: {np.average(regret_history_1)}")
# print(f"AVERAGE REGRET FOR 2: {np.average(regret_history_2)}")
# plot arm mean:
# plt.figure()
# Plots.plot_arm_means(mean_history_1, "EXP3")
plt.figure()
Plots.plot_arm_means(mean_history_2, "UCB1")
plt.legend()
plt.show()


