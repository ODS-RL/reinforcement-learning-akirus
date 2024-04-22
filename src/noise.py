import numpy as np

class BaseNoise:
    def sample(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class OUActionNoise(BaseNoise):
    # Ornstein-Uhlenbeck process (https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process)
    # Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    def __init__(self, mu, sigma = 0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)