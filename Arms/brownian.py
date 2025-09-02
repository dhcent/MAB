import numpy as np
class BrownianArm:
    def __init__(self, mu, std_dev = 0.1, drift = 0.01, volatility = 0.02):
        self.name = "BrownianArm"
        self.mean = mu
        self.std_dev = std_dev
        self.drift = drift
        self.volatility = volatility

    def simulate_brownian(self, dt = 1):
        #note, BM, variance is t, so std_dev is sqrt(t)
        dW = np.random.normal(0, np.sqrt(dt))
        self.mean = self.mean + (self.drift * dt + self.volatility * dW)

        #ONLY IF WE WANT TO BOUND
        # #bounded mean between 0 and 1
        # if self.mean < 0:
        #     self.mean = 0
        # elif self.mean > 1:
        #     self.mean = 1

    def draw(self, dt=1):
        #updates mean based on brownian motion
        #self.simulate_brownian(dt) honestly should call simulate_brownian in main
        #bounds reward between 0 and 1
        #oracle picks from all the arms draw val **
        return np.random.normal(self.mean, self.std_dev)