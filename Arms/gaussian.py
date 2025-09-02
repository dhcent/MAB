import numpy as np
class GaussianArm:
    def __init__(self, mu, std_dev=0.1):
        self.name = "GaussianArm"
        self.mean = mu
        self.std_dev = std_dev

    def update_val(self, mu):
        self.mean = mu
        return

    def draw(self):
        return np.random.normal(self.mean, self.std_dev)