import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class Environment:
    def __init__(self) -> None:
        self.shape = (10, 10)

        self.rewards = np.full(self.shape, -1)
        # The goal is marked with a 100
        self.rewards[-1, -1] = 100

        # The walls are marked with - 100
        self.rewards[2:4, 2:4] = -100
        self.rewards[5:8, 5:8] = -100
        self.rewards[8:10, 1:3] = -100

        self.actions = ("up", "down", "left", "right")

        self.state = (0, 0)

    @property
    def terminated(self) -> bool:
        if self.rewards[self.state[0], self.state[1]] in [-100, 100]:
            return True
        else:
            return False

    def reset(self) -> None:
        self.state = (0, 0)

        return self.state

    def step(self, action: Union[int, str]) -> None:
        action = action if isinstance(action, str) else self.actions[action]

        if action == "up" and self.state[0] > 0:
            self.state = (self.state[0] - 1, self.state[1])
        elif action == "down" and self.state[0] < self.shape[0] - 1:
            self.state = (self.state[0] + 1, self.state[1])
        elif action == "left" and self.state[1] > 0:
            self.state = (self.state[0], self.state[1] - 1)
        elif action == "right" and self.state[1] < self.shape[1] - 1:
            self.state = (self.state[0], self.state[1] + 1)

        reward = self.rewards[self.state[0], self.state[1]]

        return self.state, reward, self.terminated

    def render(self, show: bool = True, save_path: str = None) -> None:
        render_space = np.copy(self.rewards)

        render_space[self.state] = 50

        plt.imshow(render_space, cmap="plasma")

        if show:
            plt.show()
        if save_path:
            plt.axis("off")
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
