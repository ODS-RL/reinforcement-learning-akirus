# Run with `python3 -m examples.train_snake_dqn`
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn.functional as F


from tqdm import tqdm
from src.envs import SnakeGameEnvironment

env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def greedy_policy(model, state: tuple[int, int]) -> int:
    state = torch.tensor(state, dtype=torch.float)
    return torch.argmax(model(state))
  

def epsilon_greedy_policy(
    model, state: tuple[int, int], epsilon: float
) -> int:
    """Epsilon-greedy Action Selection"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(env.actions))
    else:
        return greedy_policy(model, state)
    


n_episodes = 300
learning_rate = 0.01#0.005
gamma = 0.95  # Discount factor
epsilon = 0.1
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.001


model = Model(11, 128, 3)

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = MSELoss()


def train(
    model,
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
    tqdm_range = tqdm(range(n_episodes), total=n_episodes)

    losses = []
    for episode in tqdm_range:
        state = env.reset()

        epsilon = (
            max_epsilon - (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            if decay_epsilon
            else epsilon
        )

        episode_losses = []
        while True:
            action = epsilon_greedy_policy(model, state, epsilon)

            next_state, reward, terminated = env.step(action)
            max_score = max(max_score, env.score)

            prediction = model(torch.tensor(state, dtype=torch.float, requires_grad=True)) # size = 3
            target = prediction.clone()
            
            if not terminated:
                q = reward + gamma * torch.max(model(torch.tensor(next_state, dtype=torch.float, requires_grad=True)))
            else:
                q = reward

            target[action] = q

            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            episode_losses.append(loss.item())

            if terminated:
                break

            state = next_state

        losses.append(np.mean(episode_losses))
        tqdm_range.set_description(f"Loss: {losses[-1]:.3f}")

    print(f"Max Score: {max_score}")

    return losses



train(
    model,
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
torch.save(model.state_dict(), "saves/snake_dqn.pt")


  

def test(model: torch.nn.Module):
    env.speed = 10
    state = env.reset()
    terminated = False
    while not terminated:
        action = greedy_policy(model, state)
        state, reward, terminated = env.step(action)



test(model)