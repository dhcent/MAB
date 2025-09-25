#oracle algorithm. Always picks the best arm.
import numpy as np


class Oracle:
    def __init__(self, arms):
        #counts is # of times each arm was visited
        self.name = "Oracle"

    # #reinitialize count and num of arms
    # def reinitialize(self, arms):
    #     self.arms = arms
    #     return

    def select_arm(self, pulled_rewards):
        # expected_vals = [arm.mean for arm in self.arms]
        return pulled_rewards.argmax()


