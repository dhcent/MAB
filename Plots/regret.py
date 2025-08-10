import matplotlib.pyplot as plt
import numpy as np
def plot_regret(regret_history):
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Number of Rounds, T")
    for t in range(len(regret_history)):
        plt.plot(t, regret_history[t], 'bo')
    plt.show()

def plot_regret_log_t(regret_history):
    plt.ylabel("Cumulative Regret")
    plt.xlabel("log T")
    for t in range(len(regret_history)):
        plt.plot(np.log(t), regret_history[t], 'bo')
    plt.show()