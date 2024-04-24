# Run with `python3 -m examples.train_cartpole_vpg`
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from src.trainers import VPGTrainer
import gymnasium as gym


env = gym.make("CartPole-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = VPGTrainer(env, device=device) # With memory




n_episodes = 500
gamma = 0.99  # Discount factor
learning_rate = 1e-3

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)
    
actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = Adam(actor.parameters(), lr=learning_rate)
    

trainer.train(
    actor = actor,
    optimizer = optimizer,
    n_episodes = n_episodes,
    gamma = gamma,
    batch_size=32
)
    
env.close()


env = gym.make("CartPole-v1", render_mode="human")

def test(model: torch.nn.Module):
    state, info = env.reset()
    terminated = False
    while not terminated:
        # action = trainer.policy(model, state)
        action = torch.argmax(model(torch.tensor(state, dtype=torch.float32).to(device)))

        state, reward, terminated, trancated, _ = env.step(action.item())
        env.render()

test(actor)

env.close()

