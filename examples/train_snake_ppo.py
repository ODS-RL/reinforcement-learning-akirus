# Run with `python3 -m examples.train_snake_ppo`
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from src.envs import SnakeGameEnvironment
from src.trainers import PPOTrainer


env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)


trainer = PPOTrainer(env)

n_episodes = 1000
learning_rate = 0.005
gamma = 0.99
batch_size = 512
hidden_size = 128

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

actor = Actor(11, hidden_size, 3)
critic = Critic(11, hidden_size)

actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=learning_rate)

trainer.train(
    actor=actor,
    critic=critic,
    criterion=nn.MSELoss(),
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    n_episodes=n_episodes,
    batch_size=batch_size,
    ppo_epochs=4,
    gamma=gamma,
    clip_param=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    max_steps=1000
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(actor.state_dict(), "saves/snake_ppo_actor.pt")
torch.save(critic.state_dict(), "saves/snake_ppo_critic.pt")


def test(actor: torch.nn.Module):
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    while not terminated:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor(state_tensor)
            action = torch.argmax(probs).item()
        state, reward, terminated, _, _ = env.step(action)

test(actor)