import gymnasium as gym
from typing import Union


class BaseEnvironment(gym.Env):
    def __init__(self) -> None:
        self.state = None
        self.actions = None

    @property
    def terminated(self) -> bool:
        raise NotImplementedError("Subclasses must implement terminated property.")

    def reset(self) -> None:
        raise NotImplementedError("Subclasses must implement reset method.")

    def step(self, action: Union[int, str]) -> tuple:
        raise NotImplementedError("Subclasses must implement step method.")

    def render(self, save_path: str = None) -> None:
        raise NotImplementedError("Subclasses must implement render method.")

    def dump(self):
        raise NotImplementedError("Subclasses must implement dump method.")
