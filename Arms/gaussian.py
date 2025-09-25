import numpy as np
class GaussianArm:
    def __init__(self, mu, std_dev=0.1, drift=0):
        self.name = "GaussianArm"
        self.mean = mu
        self.std_dev = std_dev
        self.drift = drift

    def update_val(self, mu):
        self.mean = mu
        return


    def draw(self):
        randomized_val = np.random.normal(self.mean, self.std_dev)
        self.mean += self.drift
        return randomized_val