#oracle algorithm. Always picks the best arm.

class Oracle:
    def __init__(self, arms):
        #counts is # of times each arm was visited
        self.name = "Oracle"
        self.arms = arms

    #reinitialize count and num of arms
    def reinitialize(self, arms):
        self.arms = arms
        return

    def select_arm(self):
        expected_vals = [arm.mean for arm in self.arms]
        return expected_vals.index(max(expected_vals))


