# Run with `python3 -m examples.train_snake_grpo`
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from src.envs import SnakeGameEnvironment
from src.trainers import GRPOTrainer


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


n_episodes = 8
n_steps = 32
group_size = 8
hidden_size = 128
learning_rate = 0.005
gamma = 0.99


actor = Actor(11, hidden_size, 3)

actor_optimizer = Adam(actor.parameters(), lr=learning_rate)

env = SnakeGameEnvironment(
    width=200, height=200, block_size=20, speed=10000, render_score=False
)

trainer = GRPOTrainer(env, path_clips="videos", render_epochs=[0, -1], render_limit=16)

trainer.train(
    actor=actor,
    actor_optimizer=actor_optimizer,
    n_epochs=n_episodes,
    n_steps=n_steps,
    group_size=group_size,
    kl_coeff=0.2,
    entropy_coeff=0.01,
    gamma=gamma,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/snake_grpo_actor.pt")


def test(actor: torch.nn.Module):
    env.speed = 10
    env.render_score = True
    state, _ = env.reset()
    terminated = False
    while not terminated:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor(state_tensor)
            action = torch.argmax(probs).item()
        state, reward, terminated, _, _ = env.step(action)  # type: ignore


test(actor)
