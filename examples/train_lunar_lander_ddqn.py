# Run with `python3 -m examples.train_lunar_lander_ddqn`
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from src import DDQNTrainer, ReplayMemory
import gymnasium as gym

env = gym.make("LunarLander-v2",)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# trainer = DDQNTrainer(env) # Without memory  (uncomment this line)
trainer = DDQNTrainer(env, memory=ReplayMemory(memory_size=int(1e5), batch_size=64), device=device) # With memory

n_episodes = 600
learning_rate = 5e-4#0.005
gamma = 0.99  # Discount factor
epsilon = 0.1
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.0005
tau=1e-3

class Model(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

policy_model = Model(n_observations, n_actions).to(device)
target_model = Model(n_observations, n_actions).to(device)

target_model.load_state_dict(policy_model.state_dict())

optimizer = Adam(policy_model.parameters(), lr=learning_rate)
criterion = MSELoss()


trainer.train(
    online_model= policy_model,
    target_model = target_model,
    optimizer = optimizer,
    criterion = criterion,
    n_episodes = n_episodes,
    epsilon = epsilon,
    max_epsilon = max_epsilon,
    min_epsilon = min_epsilon,
    decay_rate = decay_rate,
    decay_epsilon = "linear",
    gamma = gamma,
    max_steps=1000,
    tau = tau
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(policy_model.state_dict(), "saves/snake_cartpole_dqn.pt")
env.close()

env = gym.make("LunarLander-v2", render_mode="human")

def test(model: torch.nn.Module):
    state, info = env.reset()
    terminated = False
    while not terminated:
        action = trainer.greedy_policy(model, state)
        state, reward, terminated, trancated, _ = env.step(action.item())
        env.render()

test(policy_model)