import pygame
import random
import numpy as np
from gymnasium import spaces
from src.envs.base_env import BaseEnvironment
# Additional resources
# https://cs230.stanford.edu/projects_fall_2021/reports/103085287.pdf
# https://ceasjournal.com/index.php/CEAS/article/download/13/10)


class SnakeGameEnvironment(BaseEnvironment):
    def __init__(self, width, height, block_size, speed):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed

        self.head = (width // 2, height // 2)
        self.snake = [self.head]

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

        self.score = 0
        self.food = None
        self.place_food()

        self.actions = ("left", "straight", "right")
        self.action_space = spaces.Discrete(3)
        self.directions = ("up", "down", "left", "right")
        self.direction = random.choice(self.directions)

        self.reward = {"collision": -10, "food": 10}

    def check_collision(self, segment=None):
        if segment is None:
            segment = self.head
        if (
            segment[0] < 0
            or segment[0] >= self.width
            or segment[1] < 0
            or segment[1] >= self.height
        ):
            return True

        if segment in self.snake[:-1]:
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

        if (x, y) in self.snake:
            self.place_food()
        else:
            self.food = (x, y)

    def reset(self) -> None:
        info = None  # not implemented, not used (plug)
        self.score = 0
        self.head = (self.width // 2, self.height // 2)
        self.snake = [self.head]
        self.direction = random.choice(self.directions)
        self.place_food()

        return self.state, info

    @property
    def state(self) -> None:
        """
        States logic was borrowed from
        https://cs230.stanford.edu/projects_fall_2021/reports/103085287.pdf
        """
        up_segment = (self.head[0], self.head[1] - self.block_size)
        down_segment = (self.head[0], self.head[1] + self.block_size)
        left_segment = (self.head[0] - self.block_size, self.head[1])
        right_segment = (self.head[0] + self.block_size, self.head[1])

        return tuple(
            np.array(
                [
                    # Danger is on the straight
                    (self.check_collision(up_segment) and self.direction == "up")
                    or (self.check_collision(down_segment) and self.direction == "down")
                    or (self.check_collision(left_segment) and self.direction == "left")
                    or (
                        self.check_collision(right_segment)
                        and self.direction == "right"
                    ),
                    # Danger is on the right
                    (self.check_collision(up_segment) and self.direction == "left")
                    or (
                        self.check_collision(down_segment) and self.direction == "right"
                    )
                    or (self.check_collision(left_segment) and self.direction == "down")
                    or (self.check_collision(right_segment) and self.direction == "up"),
                    # Danger is on the left
                    (self.check_collision(up_segment) and self.direction == "right")
                    or (self.check_collision(down_segment) and self.direction == "left")
                    or (self.check_collision(left_segment) and self.direction == "up")
                    or (
                        self.check_collision(right_segment) and self.direction == "down"
                    ),
                    # Food location
                    self.head[0] < self.food[0],
                    self.head[0] > self.food[0],
                    self.head[1] < self.food[1],
                    self.head[1] > self.food[1],
                    # Direction
                    self.direction == "left",
                    self.direction == "right",
                    self.direction == "up",
                    self.direction == "down",
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
        truncated = False  # not implemented, not used (plug)
        info = None  # not implemented, not used (plug)

        self.handle_input(action)
        self.update_head()

        self.snake.append(self.head)

        if self.check_collision():
            reward = self.reward["collision"]
            terminated = True

            return (self.state, reward, terminated, truncated, info)

        if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
            reward = self.reward["food"]

            self.place_food()
            self.score += 1
        else:
            self.snake.pop(0)

        self.render()
        self.clock.tick(self.speed)

        return (self.state, reward, terminated, truncated, info)

    def handle_input(self, action: int | str):  # 0 = left, 1 = straight, 2 = right
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        action = action if isinstance(action, str) else self.actions[action]

        if self.direction == "right":
            if action == "left":
                self.direction = "up"
            elif action == "right":
                self.direction = "down"

        elif self.direction == "left":
            if action == "left":
                self.direction = "down"
            elif action == "right":
                self.direction = "up"

        elif self.direction == "up":
            if action == "left":
                self.direction = "left"
            elif action == "right":
                self.direction = "right"

        elif self.direction == "down":
            if action == "left":
                self.direction = "right"
            elif action == "right":
                self.direction = "left"

    def update_head(self):
        if self.direction == "right":
            self.head = (self.head[0] + self.block_size, self.head[1])
        elif self.direction == "left":
            self.head = (self.head[0] - self.block_size, self.head[1])
        elif self.direction == "up":
            self.head = (self.head[0], self.head[1] - self.block_size)
        elif self.direction == "down":
            self.head = (self.head[0], self.head[1] + self.block_size)

    def render(self, save_path=None):
        self.screen.fill(self.colors["beige"])
        pygame.draw.rect(
            self.screen,
            self.colors["purple"],
            [self.food[0], self.food[1], self.block_size, self.block_size],
        )

        for segment in self.snake:
            pygame.draw.rect(
                self.screen,
                self.colors["light green"],
                [segment[0], segment[1], self.block_size, self.block_size],
            )

        value = self.font.render("Score: " + str(self.score), True, self.colors["navy"])
        self.screen.blit(value, [0, 0])
        pygame.display.flip()

        if save_path is not None:
            pygame.image.save(self.screen, save_path)


class SnakeGame(SnakeGameEnvironment):
    def run(self):
        while True:
            self.handle_input()
            self.update_head()

            self.snake.append(self.head)

            if self.check_collision():
                break  # FIXME: edit this

            if self.head[0] == self.food[0] and self.head[1] == self.food[1]:
                self.place_food()
                self.score += 1
            else:
                self.snake.pop(0)

            self.render()
            self.clock.tick(self.speed)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != "right":
                    self.direction = "left"
                elif event.key == pygame.K_RIGHT and self.direction != "left":
                    self.direction = "right"
                elif event.key == pygame.K_UP and self.direction != "down":
                    self.direction = "up"
                elif event.key == pygame.K_DOWN and self.direction != "up":
                    self.direction = "down"


if __name__ == "__main__":
    snake_game = SnakeGame(width=240, height=240, block_size=20, speed=2)
    snake_game.run()
