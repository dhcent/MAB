# class BernoulliArm:
#     def __init__(self, p):
#         self.p = p
#
#     # Reward system based on Bernoulli (copied from:
#     # https://github.com/kfoofw/bandit_simulations/blob/master/python/multiarmed_bandits/analysis/eps-greedy.md)
#     def draw(self):
#         if rand.random() > self.p:
#             return 0
#         else:
#             return 1
#do this later