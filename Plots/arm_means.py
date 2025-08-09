import matplotlib.pyplot as plt

#incase one is interested in examining one particular arm.
def plot_arm_mean(arm_mean_history):
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Number of Rounds, T")
    for t in range(len(arm_mean_history)):
        plt.plot(t, arm_mean_history[t], 'bo')
    plt.show()

def plot_arm_means(arms_mean_history):
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Number of Rounds, T")
    for t in range(len(regret_history)):
        plt.plot(t, regret_history[t], 'bo')
    plt.show()
