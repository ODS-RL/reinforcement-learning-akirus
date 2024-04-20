# Run with `python3 -m examples.train_snake_ql`
import os
import numpy as np
from src import SnakeGameEnvironment
from src import SARSATrainer

env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)
    
trainer = SARSATrainer(env)

n_episodes = 1000
learning_rate = 0.01
gamma = 0.95  # Discount factor
epsilon = 0.1



q_table = np.zeros((2,) * len(env.state) + (len(env.actions),))

q_table = trainer.train(
    q_table = q_table,
    n_episodes = n_episodes,
    epsilon = epsilon,
    learning_rate = learning_rate,
    gamma = gamma,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
with open('saves/snake_sarsa.npy', 'wb') as f:
    np.save(f, q_table)

def test(q_table):
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    while not terminated:
        action = trainer.greedy_policy(q_table, state)
        state, reward, terminated, _, _ = env.step(action)
        env.render()


test(q_table)