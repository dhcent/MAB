import matplotlib.pyplot as plt

def plot_regret(regret_history):
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Number of Rounds, T")
    for t in range(len(regret_history)):
        plt.plot(t, regret_history[t], 'bo')
    plt.show()
