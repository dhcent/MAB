import matplotlib.pyplot as plt

#incase one is interested in examining one particular arm.
def plot_arm_mean(arm_mean_history, color, arm_index):
        plt.plot(range(len(arm_mean_history)),
                 arm_mean_history,
                 label = f"Arm: {arm_index}"
                 )

#input array of arm means
def plot_arm_means(arm_means_history, algo_name):
    plt.title(algo_name)
    plt.ylabel("Mean Reward")
    plt.xlabel("Number of Rounds, T")

    style = ['bo-', 'ro-', 'go-', 'co-', 'mo-', 'yo-', 'ko-']

    for i in range(len(arm_means_history)):
        plot_arm_mean(arm_means_history[i], style[i % len(style)], i)
    plt.legend()
    plt.show()
