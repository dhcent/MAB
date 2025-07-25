import numpy as np
class BrownianArm:
    def __init__(self, mu, std_dev = 0.1, drift = 0.01, volatility = 0.02):
        self.mean = mu
        self.std_dev = std_dev
        self.drift = drift
        self.volatility = volatility

    def simulate_brownian(self, steps = 1):
        #note, BM, variance is t, so std_dev is sqrt(t)
        dW = np.random.normal(0, np.sqrt(steps))
        self.mean = self.mean + (self.drift * steps + self.volatility * dW)

    def draw(self, dt=1):
        #updates mean based on brownian motion
        self.simulate_brownian(dt)
        return np.random.normal(self.mean, self.std_dev)