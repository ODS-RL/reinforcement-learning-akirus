# Run with `python3 -m examples.train_pendulum_ddpg`
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from src.trainers import DDPGTrainer
from src.buffers import ReplayMemory
from src.noise import OUActionNoise
import gymnasium as gym

env = gym.make("Pendulum-v1")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = DDPGTrainer(env, memory=ReplayMemory(memory_size=int(1e5), batch_size=128), device=device)

n_episodes = 300
actor_learning_rate = 5e-4
critic_learning_rate = 5e-3
gamma = 0.99  # Discount factor
tau=1e-2

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, scale = env.action_space.high[0]):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
        self.scale = scale
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.layer3(x)) * self.scale
    
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.shape[0] # For continuous (non-discrete) action space


actor = Actor(n_observations, n_actions).to(device)
critic = Critic(n_observations, n_actions).to(device)

actor_target = Actor(n_observations, n_actions).to(device)
critic_target = Critic(n_observations, n_actions).to(device)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())


actor_optimizer = Adam(actor.parameters(), lr=actor_learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=critic_learning_rate)

criterion = MSELoss()

ou_noise = OUActionNoise(mu = np.zeros(env.action_space.shape[0]))

trainer.train(
    actor=actor,
    critic=critic,
    actor_target=actor_target,
    critic_target=critic_target,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    criterion = criterion,
    noise=ou_noise,
    n_episodes = n_episodes,
    gamma = gamma,
    max_steps=1000,
    tau = tau
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/pendulum_ddpg.pt")
env.close()

env = gym.make("Pendulum-v1", render_mode="human")

def test(model: torch.nn.Module):
    state, info = env.reset()
    terminated = False
    while not terminated:
        action = trainer.policy(model, state)
        state, reward, terminated, trancated, _ = env.step([action.item()])
        env.render()

test(actor)

env.close()