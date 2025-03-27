# Additional resources
# https://cs230.stanford.edu/projects_fall_2021/reports/103085287.pdf
# https://ceasjournal.com/index.php/CEAS/article/download/13/10)

import random

from copy import deepcopy

import numpy as np
import pygame

from gymnasium import spaces

from src.envs.base_env import BaseEnvironment


class SnakeState:
    head: tuple | None = None
    snake: list | None = None
    food: tuple | None = None
    direction: str | None = None
    score: float | None = None
    steps: int | None = None

    def __repr__(self) -> str:
        return f"{self.steps, self.food, self.direction, self.head, self.snake, self.score}"


class SnakeGameEnvironment(BaseEnvironment):
    def __init__(self, width, height, block_size, speed, max_steps=200):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed

        self._dump = None
        self._state = SnakeState()
        self._state.head = (width // 2, height // 2)
        self._state.snake = [self._state.head]

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "beige": (255, 245, 224),
            "light green": (65, 176, 110),
            "purple": (210, 0, 98),
            "navy": (20, 30, 70),
        }

        self._state.score = 0
        self._state.food = None
        self.place_food()
        self._state.steps = 0

        self.actions = ("left", "straight", "right")
        self.action_space = spaces.Discrete(len(self.actions))
        self.directions = ("up", "down", "left", "right")
        self._state.direction = random.choice(self.directions)  # type: ignore

        self.reward = {"collision": -10, "food": 10}
        self.max_steps = int(
            np.ceil(width * height * np.sqrt(2)).round()
        )  # TODO: use wrapper

    def check_collision(self, segment=None):
        if segment is None:
            segment = self._state.head
        if (
            segment[0] < 0
            or segment[0] >= self.width
            or segment[1] < 0
            or segment[1] >= self.height
        ):
            return True

        if segment in self._state.snake[:-1]:
            return True

        return False

    def place_food(self):
        x = (
            random.randint(0, (self.width - self.block_size) // self.block_size)
            * self.block_size
        )
        y = (
            random.randint(0, (self.height - self.block_size) // self.block_size)
            * self.block_size
        )

        if (x, y) in self._state.snake:
            self.place_food()
        else:
            self._state.food = (x, y)

    def reset(self, state: SnakeState | None = None) -> None:
        info = None  # not implemented, not used (plug)
        if isinstance(state, SnakeState):
            # Restore state
            self._state = deepcopy(state)
        else:
            # Initial state
            self._state = SnakeState()
            self._state.score = 0
            self._state.head = (self.width // 2, self.height // 2)
            self._state.snake = [self._state.head]
            self._state.direction = random.choice(self.directions)
            self.place_food()  # initialize self._state.food
            self._state.steps = 0

        return self.state, info

    def dump(self) -> SnakeState:
        return deepcopy(self._state)

    @property
    def state(self) -> None:
        """
        States logic was borrowed from
        https://cs230.stanford.edu/projects_fall_2021/reports/103085287.pdf
        """
        up_segment = (self._state.head[0], self._state.head[1] - self.block_size)
        down_segment = (self._state.head[0], self._state.head[1] + self.block_size)
        left_segment = (self._state.head[0] - self.block_size, self._state.head[1])
        right_segment = (self._state.head[0] + self.block_size, self._state.head[1])

        return tuple(
            np.array(
                [
                    # Danger is ahead
                    (self.check_collision(up_segment) and self._state.direction == "up")
                    or (
                        self.check_collision(down_segment)
                        and self._state.direction == "down"
                    )
                    or (
                        self.check_collision(left_segment)
                        and self._state.direction == "left"
                    )
                    or (
                        self.check_collision(right_segment)
                        and self._state.direction == "right"
                    ),
                    # Danger is on the right
                    (
                        self.check_collision(up_segment)
                        and self._state.direction == "left"
                    )
                    or (
                        self.check_collision(down_segment)
                        and self._state.direction == "right"
                    )
                    or (
                        self.check_collision(left_segment)
                        and self._state.direction == "down"
                    )
                    or (
                        self.check_collision(right_segment)
                        and self._state.direction == "up"
                    ),
                    # Danger is on the left
                    (
                        self.check_collision(up_segment)
                        and self._state.direction == "right"
                    )
                    or (
                        self.check_collision(down_segment)
                        and self._state.direction == "left"
                    )
                    or (
                        self.check_collision(left_segment)
                        and self._state.direction == "up"
                    )
                    or (
                        self.check_collision(right_segment)
                        and self._state.direction == "down"
                    ),
                    # Food location
                    self._state.head[0] < self._state.food[0],
                    self._state.head[0] > self._state.food[0],
                    self._state.head[1] < self._state.food[1],
                    self._state.head[1] > self._state.food[1],
                    # Direction
                    self._state.direction == "left",
                    self._state.direction == "right",
                    self._state.direction == "up",
                    self._state.direction == "down",
                ]
            ).astype(int)
        )

    def step(self, action: int | str) -> tuple[tuple[bool], int, bool, bool, dict]:
        """Make a step in the environment

        Step produces a transition [state, reward, terminated, truncated, info]

        TODO: update reward logic
        Example: https://ceasjournal.com/index.php/CEAS/article/download/13/10

        Args:
            action (int | str): 0 = left, 1 = strait, 2 = right

        Returns:
            tuple[tuple[bool], int, bool, bool, dict]: transition
        """
        reward = 0
        terminated = False
        truncated = self._state.steps >= self.max_steps  # TODO: use wrapper
        info = {}  # not implemented, not used (plug)

        self.handle_input(action)
        self.update_head()

        self._state.snake.append(self._state.head)

        if self.check_collision():
            reward = self.reward["collision"]
            terminated = True

            return (self.state, reward, terminated, truncated, info)

        if (
            self._state.head[0] == self._state.food[0]
            and self._state.head[1] == self._state.food[1]
        ):
            reward = self.reward["food"]

            self.place_food()
            self._state.score += 1
        else:
            self._state.snake.pop(0)

        info["frame"] = self.render()
        try:
            info["frame"] = np.transpose(
                info["frame"],  # type: ignore
                (1, 0, 2),
            )  # [W, H, C] -> [H, W, C]
        except ValueError:
            ...
        self.clock.tick(self.speed)

        return (self.state, reward, terminated, truncated, info)

    def handle_input(self, action: int | str):  # 0 = left, 1 = straight, 2 = right
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        action = action if isinstance(action, str) else self.actions[action]

        if self._state.direction == "right":
            if action == "left":
                self._state.direction = "up"
            elif action == "right":
                self._state.direction = "down"

        elif self._state.direction == "left":
            if action == "left":
                self._state.direction = "down"
            elif action == "right":
                self._state.direction = "up"

        elif self._state.direction == "up":
            if action == "left":
                self._state.direction = "left"
            elif action == "right":
                self._state.direction = "right"

        elif self._state.direction == "down":
            if action == "left":
                self._state.direction = "right"
            elif action == "right":
                self._state.direction = "left"

    def update_head(self):
        if self._state.direction == "right":
            self._state.head = (
                self._state.head[0] + self.block_size,
                self._state.head[1],
            )
        elif self._state.direction == "left":
            self._state.head = (
                self._state.head[0] - self.block_size,
                self._state.head[1],
            )
        elif self._state.direction == "up":
            self._state.head = (
                self._state.head[0],
                self._state.head[1] - self.block_size,
            )
        elif self._state.direction == "down":
            self._state.head = (
                self._state.head[0],
                self._state.head[1] + self.block_size,
            )

    def render(self, save_path=None):
        self._state.steps += 1
        frame = None

        self.screen.fill(self.colors["beige"])
        pygame.draw.rect(
            self.screen,
            self.colors["purple"],
            [
                self._state.food[0],
                self._state.food[1],
                self.block_size,
                self.block_size,
            ],
        )

        for segment in self._state.snake:
            pygame.draw.rect(
                self.screen,
                self.colors["light green"],
                [segment[0], segment[1], self.block_size, self.block_size],
            )

        value = self.font.render(
            "Score: " + str(self._state.score), True, self.colors["navy"]
        )
        self.screen.blit(value, [0, 0])
        pygame.display.flip()

        if save_path is not None:
            pygame.image.save(self.screen, save_path)
        else:
            frame = pygame.surfarray.array3d(self.screen)
        return frame


class SnakeGame(SnakeGameEnvironment):
    def run(self):
        while True:
            self.handle_input()
            self.update_head()

            self._state.snake.append(self._state.head)

            if self.check_collision():
                break  # FIXME: edit this

            if (
                self._state.head[0] == self._state.food[0]
                and self._state.head[1] == self._state.food[1]
            ):
                self.place_food()
                self._state.score += 1
            else:
                self._state.snake.pop(0)

            self.render()
            self.clock.tick(self.speed)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self._state.direction != "right":
                    self._state.direction = "left"
                elif event.key == pygame.K_RIGHT and self._state.direction != "left":
                    self._state.direction = "right"
                elif event.key == pygame.K_UP and self._state.direction != "down":
                    self._state.direction = "up"
                elif event.key == pygame.K_DOWN and self._state.direction != "up":
                    self._state.direction = "down"
                elif event.key == pygame.K_s:
                    self._dump = self.dump()
                    print(f"DEBUG: save dump = {self._dump}")
                elif event.key == pygame.K_l:
                    if isinstance(self._dump, SnakeState):
                        # self._state = self._dump
                        self.reset(self._dump)
                        print(f"DEBUG: load dump = {self._dump}")
                    else:
                        print("WARNING: dump is not valid. Failed to load!")


if __name__ == "__main__":
    snake_game = SnakeGame(width=240, height=240, block_size=20, speed=2)
    snake_game.run()
