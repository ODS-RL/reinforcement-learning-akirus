import sys, os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

import numpy as np
from tqdm import tqdm
from src.envs import SimpleEnvironment
from PIL import Image

# Inspired by https://huggingface.co/learn/deep-rl-course/unit2/hands-on

env = SimpleEnvironment()
print(f"Rewards space:\n{env.rewards}")



def greedy_policy(q_table: np.ndarray, state: tuple[int, int]) -> int:
    return np.argmax(q_table[state])

def epsilon_greedy_policy(
    q_table: np.ndarray, state: tuple[int, int], epsilon: float
) -> int:
    """Epsilon-greedy Action Selection"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(env.actions))
    else:
        return greedy_policy(q_table, state)


n_episodes = 10000
learning_rate = 0.9
gamma = 0.9  # Discount factor
epsilon = 0.1
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.001
max_steps = 100


def train(
    q_table: np.ndarray,
    n_episodes: int,
    epsilon: float,
    max_epsilon: float,
    min_epsilon: float,
    decay_rate: float,
    decay_epsilon: bool,
    learning_rate: float,
    gamma,
) -> np.ndarray:
    for episode in tqdm(range(n_episodes), total=n_episodes):
        state = env.reset()

        epsilon = (
            max_epsilon - (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            if decay_epsilon
            else epsilon
        )

        for step in range(max_steps):
            action = epsilon_greedy_policy(q_table, state, epsilon)

            next_state, reward, terminated = env.step(action)

            # The Bellman Equation
            q_table[state][action] += learning_rate * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state

            if terminated:
                break

    return q_table


q_table = np.zeros((env.shape[0], env.shape[1], len(env.actions)))
q_table = train(
    q_table,
    n_episodes,
    epsilon,
    max_epsilon,
    min_epsilon,
    decay_rate,
    False,
    learning_rate,
    gamma,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
with open('saves/simple_ql.npy', 'wb') as f:
    np.save(f, q_table)

def test(q_table: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    state = env.reset()
    terminated = False
    total_reward = 0
    states = []

    i = 0
    while not terminated:
        env.render(show=False, save_path=f"images/render-{i}.png")

        action = greedy_policy(q_table, state)
        state, reward, terminated = env.step(action)

        total_reward += reward
        states.append(state)

        i += 1

    return total_reward, states


total_reward, states = test(q_table)

print(f"Total reward: {total_reward}")
print(f"States: {states}")

# Save render as gif
frames = [Image.open(f"images/render-{i}.png") for i in range(len(states))]
frames[0].save(
    "images/render.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=100,
    loop=0,
)
