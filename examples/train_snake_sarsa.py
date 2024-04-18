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
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.001


q_table = np.zeros((2,) * len(env.state) + (len(env.actions),))

q_table = trainer.train(
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
with open('saves/snake_sarsa.npy', 'wb') as f:
    np.save(f, q_table)

def test(q_table):
    env.speed = 10
    state = env.reset()
    terminated = False
    while not terminated:
        action = trainer.greedy_policy(q_table, state)
        state, reward, terminated = env.step(action)
        env.render()


test(q_table)