def run_algorithm(algorithm, horizon):
    total_reward = 0
    cumulative_rewards = [0] * len(arms)

    algorithm.initialize(n_arms=len(arms))
    #horizon is total # of steps
    for t in range (horizon):
        #chosen arm is index
        chosen_arm = algorithm.select_arm()
        reward = arms[chosen_arm].draw()
        algorithm.update(chosen_arm, reward)
        cumulative_rewards[chosen_arm] += reward
        print(f"{chosen_arm}: {reward}")

    return cumulative_rewards

#horizon is # of steps or rounds that alg will run
arms = []
expected_vals = [rand.random() for i in range(100)]
for val in expected_vals:
    arms.append(NormalArm(val))

    #given algorithm + horizon
eps = 0.2
horizon = 100
algo = EpsilonGreedy(eps, n_arms = len(arms))
run_algorithm(algo, horizon)

for i in range (len(arms)):
    print(f"{i}: {arms[i].expected_val}")
#print(max(arms))