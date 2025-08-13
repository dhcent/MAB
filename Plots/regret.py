import matplotlib.pyplot as plt
import numpy as np

#allow label as parameter to allow flexibility between plotting regret and cumulative regret
def plot_regret(regret_history, y_label="regret", log = False):
    plt.ylabel(y_label)
    if log:
        plt.xlabel("log T")
        for t in range(len(regret_history)):
            if t != 0:
                plt.plot(np.log(t), regret_history[t], 'bo-')
            else:
                plt.plot(0, regret_history[t], 'bo-')
    else:
        plt.xlabel("Number of Rounds, T")
        for t in range(len(regret_history)):
            plt.plot(t, regret_history[t], 'bo-')
    plt.show()

