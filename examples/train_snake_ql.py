# Run with `python3 -m examples.train_snake_ql`
import os
import numpy as np
from tqdm import tqdm
from src.envs import SnakeGameEnvironment


env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)

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
    


n_episodes = 1000
learning_rate = 0.01
gamma = 0.95  # Discount factor
epsilon = 0.1
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.001


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
    max_score = 0
    for episode in tqdm(range(n_episodes), total=n_episodes):
        state = env.reset()

        epsilon = (
            max_epsilon - (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            if decay_epsilon
            else epsilon
        )

        while True:
            action = epsilon_greedy_policy(q_table, state, epsilon)

            next_state, reward, terminated = env.step(action)
            max_score = max(max_score, env.score)
            
            # The Bellman Equation
            q_table[state][action] += learning_rate * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            if terminated:
                break

            state = next_state

    print(f"Max Score: {max_score}")
    return q_table


q_table = np.zeros((2,) * len(env.state) + (len(env.actions),))

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
with open('saves/snake_ql.npy', 'wb') as f:
    np.save(f, q_table)

def test(q_table):
    env.speed = 10
    state = env.reset()
    terminated = False
    while not terminated:
        action = greedy_policy(q_table, state)
        state, reward, terminated = env.step(action)
        env.render()


test(q_table)