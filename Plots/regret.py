import matplotlib.pyplot as plt
import numpy as np

#allow label as parameter to allow flexibility between plotting regret and cumulative regret
def plot_regret(regret_history, algo_name, y_label="Regret", log = False, color='blue'):
    plt.title(f"{y_label} Comparison")
    plt.ylabel(y_label)
    if log:
        plt.xlabel("log T")
        #+1 case since np.log(0) is undefined
        x_vals = np.log(np.arange(1, len(regret_history) + 1))
    else:
        plt.xlabel("T")
        x_vals = np.arange(len(regret_history))

    plt.plot(x_vals, regret_history, label=algo_name, color=color, linewidth=1, linestyle="-")
    plt.legend()
    #don't plt.show here, do that in main

