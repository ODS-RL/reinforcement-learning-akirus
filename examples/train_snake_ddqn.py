# Run with `python3 -m examples.train_snake_ddqn`
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss
from src.envs import SnakeGameEnvironment
from src.trainers import DDQNTrainer
from src.buffers import ReplayMemory
env = SnakeGameEnvironment(
    width=200,
    height=200,
    block_size=20,
    speed=10000
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# trainer = DDQNTrainer(env, device = device) # Without memory(uncomment this line)
trainer = DDQNTrainer(env, memory=ReplayMemory(memory_size=256, batch_size=8), device=device) # With memory 

n_episodes = 300
learning_rate = 0.01#0.005
gamma = 0.95  # Discount factor
min_epsilon = 0.01
max_epsilon = 1
decay_rate = 0.05 

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

policy_model = Model(11, 128, 3).to(device)
target_model = Model(11, 128, 3).to(device)
target_model.load_state_dict(policy_model.state_dict())

optimizer = Adam(policy_model.parameters(), lr=learning_rate)
criterion = MSELoss()


scores = trainer.train(
    online_model= policy_model,
    target_model = target_model,
    optimizer = optimizer,
    criterion = criterion,
    n_episodes = n_episodes,
    max_epsilon = max_epsilon,
    min_epsilon = min_epsilon,
    decay_rate = decay_rate,
    decay_epsilon = "exponential",
    gamma = gamma,
)

if not os.path.exists("saves"):
    os.makedirs("saves")
torch.save(policy_model.state_dict(), "saves/snake_ddqn.pt")


  

def test(model: torch.nn.Module):
    env.speed = 10
    state, _ = env.reset()
    terminated = False
    while not terminated:
        action = trainer.greedy_policy(model, state)
        state, reward, terminated, _, _ = env.step(action)

test(policy_model)