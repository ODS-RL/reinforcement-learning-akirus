import numpy as np
import torch
from tqdm import tqdm
from src.envs.base_env import BaseEnvironment

class QLearningTrainer:
    def __init__(self, env: BaseEnvironment) -> None:
        self.env = env

    def greedy_policy(self, q_table: np.ndarray, state: tuple[int, int]) -> int:
        return np.argmax(q_table[state])

    def epsilon_greedy_policy(self,
        q_table: np.ndarray, state: tuple[int, int], epsilon: float
    ) -> int:
        """Epsilon-greedy Action Selection"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(len(self.env.actions))
        else:
            return self.greedy_policy(q_table, state)

    def train(
        self,
        q_table: np.ndarray,
        n_episodes: int,
        epsilon: float,
        max_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        decay_epsilon: bool,
        learning_rate: float,
        gamma,
        max_steps: int = None,
    ) -> np.ndarray:
        for episode in tqdm(range(n_episodes), total=n_episodes):
            state = self.env.reset()

            epsilon = (
                max_epsilon - (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                if decay_epsilon
                else epsilon
            )

            step = 0
            while True:
                action = self.epsilon_greedy_policy(q_table, state, epsilon)

                next_state, reward, terminated = self.env.step(action)

                # The Bellman Equation
                q_table[state][action] += learning_rate * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
                )

                state = next_state

                if terminated or step == max_steps:
                    break

                step += 1

        return q_table

class DQNTrainer:
    def __init__(self, env: BaseEnvironment) -> None:
        self.env = env

    def greedy_policy(self, model, state: tuple[int, int]) -> int:
        state = torch.tensor(state, dtype=torch.float)
        return torch.argmax(model(state))    

    def epsilon_greedy_policy(self,
        model, state: tuple[int, int], epsilon: float
    ) -> int:
        """Epsilon-greedy Action Selection"""
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(len(self.env.actions))
        else:
            return self.greedy_policy(model, state)
        
    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        n_episodes: int,
        epsilon: float,
        max_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        decay_epsilon: bool,
        gamma,
        max_steps: int = None,
    ) -> np.ndarray:
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        losses = []
        for episode in tqdm_range:
            state = self.env.reset()

            epsilon = (
                max_epsilon - (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                if decay_epsilon
                else epsilon
            )

            episode_losses = []
            step = 0
            while True:
                action = self.epsilon_greedy_policy(model, state, epsilon)

                next_state, reward, terminated = self.env.step(action)

                prediction = model(torch.tensor(state, dtype=torch.float, requires_grad=True))
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

                state = next_state

                if terminated or step == max_steps:
                    break

                step += 1
                

            losses.append(np.mean(episode_losses))
            tqdm_range.set_description(f"Loss: {losses[-1]:.3f}")

        return losses


class SarsaTrainer: # TODO: Implement this
    def __init__(self):
        pass 


