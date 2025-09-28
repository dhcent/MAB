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
def run_algorithm(algorithm, horizon, arms, state_informed, epoch = 0):
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
        #!!!!!!!!!!!!!! FOR UCB THE BOUND IS STATE OBLIVIOUS
        chosen_arm_index = algorithm.select_arm()

        if state_informed:
            algorithm.update(chosen_arm_index, arms[chosen_arm_index].mean)  # STATE INFORMED CASE:
        else:
            reward = pulled_rewards[chosen_arm_index]   # STATE OBLIVIOUS CASE:
            algorithm.update(chosen_arm_index, reward)

        #oracle action for pseudo-regret
        oracle_chosen_arm_index = oracle.select_arm(arm_means)

        # note: this is pseudo-regret
        regret = arms[oracle_chosen_arm_index].mean - arms[chosen_arm_index].mean
        #for steady-state regret
        regret_history.append(regret)

        #used for plotting arm means
        for i in range(len(arms)):
            mean_history[i].append(arms[i].mean)

    #cumulative regret
    cumulative_regret_history = np.cumsum(regret_history)

    #for calculating avg regret (steady state regret)
    avg_regret = np.average(regret_history)

    return regret_history, cumulative_regret_history, mean_history, avg_regret
#horizon is # of steps or rounds that alg will run


def generate_arms():
    k = 5 #num of arms
    means = [rand.random() for i in range(k)]
    arms = []
    # generate each arm based on Arm Type and k # of arms
    std_dev = 0.05
    drift = 0
    volatility = 0.75

    #4 gaussian, 1 brownian
    arms.append(Arms.GaussianArm(means[0], std_dev))
    # arms.append(Arms.GaussianArm(means[1], std_dev))
    # arms.append(Arms.GaussianArm(means[2], std_dev))
    # arms.append(Arms.GaussianArm(means[3], std_dev))

    #brownian arms
    arms.append(Arms.BrownianArm(means[1], std_dev, drift, volatility))
    arms.append(Arms.BrownianArm(means[2], std_dev, drift, volatility))
    arms.append(Arms.BrownianArm(means[3], std_dev, drift, volatility))
    arms.append(Arms.BrownianArm(means[4], std_dev, drift, volatility))
    return arms, means

#NOTE: GENERATED ARMS ARE DIFFERENT WITH THIS CURRENT IMPLEMENTATION
# arms1 = generate_arms(arm_type)
arms2, expected_vals = generate_arms()

for i in range(len(arms2)):
    print(round(arms2[i].mean, 5))

#given algorithm + horizon
eps = 0.03
horizon = 8000
epoch = 10000

# eps_algo = EpsilonGreedy(eps, n_arms = len(arms1))
UCB_algo = UCB1(n_arms = len(arms2))
#EXP3(gamma, n_arms)
#EXP3_algo = EXP3(0.1, len(arms1))
# regret_history_1, cumulative_regret_history_1, mean_history_1 = run_algorithm(EXP3_algo, horizon, arms1, epoch)
informed = False
regret_history_2, cumulative_regret_history_2, mean_history_2, steady_state_regret = run_algorithm(UCB_algo, horizon, arms2, informed, epoch)

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

#k = # of arms
def calc_slivkins_theoretical_steadystate_regret_bound(k, avg_volatility):
    return k * avg_volatility * np.log(1 / avg_volatility)

def calc_slivkins_phase_length(avg_volatility):
    #CHECK IF LOGARITHMIC FUNCTION SHOULD BE BASE E OR BASE 10
    return  (1 /avg_volatility) * np.sqrt(np.log(1 / avg_volatility))

UCB_bound = calc_UCB_theoretical_regret_bound(horizon, expected_vals)
print(f"Theoretical UCB Regret Upperbound: {round(UCB_bound, 5)}")
print(f"Actual Cumulative Regret: {round(cumulative_regret_history_2[-1], 5)}")
print(f"Steady State Regret: {steady_state_regret}")

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


