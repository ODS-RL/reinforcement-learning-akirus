# Run with `python3 -m examples.train_cartpole_vpg`
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss
from src.trainers import A2CTrainer #Discrete version
import gymnasium as gym


env = gym.make("CartPole-v1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = A2CTrainer(env, device=device)




n_episodes = 500
gamma = 0.99  # Discount factor
learning_rate = 1e-3

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)
    

class Critic(nn.Module):
    def __init__(self, n_observations):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    

    
actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
critic = Critic(env.observation_space.shape[0]).to(device)

actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=learning_rate)
    

trainer.train(
    actor = actor,
    critic = critic,
    actor_optimizer = actor_optimizer,
    critic_optimizer = critic_optimizer,
    criterion=MSELoss(),
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

