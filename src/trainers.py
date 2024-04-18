import numpy as np
import torch
import random
from collections import deque
from tqdm import tqdm
from src.envs.base_env import BaseEnvironment

# https://arxiv.org/pdf/1312.5602.pdf
# https://arxiv.org/pdf/1509.02971.pdf


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
        gamma: float,
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



class ReplayMemory:
    def __init__(self, memory_size: int, batch_size: int) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=memory_size)

    def add(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class DQNTrainer:
    def __init__(self, env: BaseEnvironment, memory: ReplayMemory = None) -> None:
        self.env = env
        self.memory = memory

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
        gamma: float,
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

                if self.memory is not None:
                    self.memory.add(state, action, next_state, reward, terminated)

                    if len(self.memory) >= self.memory.batch_size:
                        transitions = self.memory.sample()

                        for transition in transitions:
                            loss = self._train(model, optimizer, criterion, gamma, transition)

                            episode_losses.append(loss.item())
                
                else:
                    loss = self._train(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        gamma=gamma,
                        transition=(state, action, next_state, reward, terminated),
                    )

                    episode_losses.append(loss.item())

                state = next_state

                if terminated or step == max_steps:
                    break

                step += 1
                

            losses.append(np.mean(episode_losses))
            tqdm_range.set_description(f"Loss: {losses[-1]:.3f}")

        return losses
    
    def _train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transition,
    ):
        state, action, next_state, reward, terminated = transition
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

        return loss


class DDQNTrainer:
    """Double DQN Trainer"""
    # https://arxiv.org/pdf/1509.06461.pdf
    def __init__(self, env: BaseEnvironment, memory: ReplayMemory = None) -> None:
        self.env = env
        self.memory = memory

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
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        n_episodes: int,
        epsilon: float,
        max_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
        decay_epsilon: bool,
        gamma: float,
        tau = 0.005,
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
                action: int = self.epsilon_greedy_policy(online_model, state, epsilon)

                next_state, reward, terminated = self.env.step(action)

                if self.memory is not None:
                    self.memory.add(state, action, next_state, reward, terminated)

                    if len(self.memory) > self.memory.batch_size:
                        transitions = self.memory.sample()

                        for transition in transitions:
                            loss = self._train(online_model, target_model, optimizer, criterion, gamma, transition)

                            episode_losses.append(loss.item())

                else:
                    loss = self._train(
                        online_model=online_model,
                        target_model=target_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        gamma=gamma,
                        transition=(state, action, next_state, reward, terminated),
                    )

                    episode_losses.append(loss.item())

                target_weights = target_model.state_dict()
                policy_weights = online_model.state_dict()
                for key in policy_weights:
                    target_weights[key] = policy_weights[key]*tau + target_weights[key]*(1-tau)
                target_model.load_state_dict(target_weights)

                state = next_state

                if terminated or step == max_steps:
                    break

                step += 1

            losses.append(np.mean(episode_losses))
            tqdm_range.set_description(f"Loss: {losses[-1]:.3f}")

        return losses
    
    def _train(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transition,
    ):
        state, action, next_state, reward, terminated = transition
        prediction = online_model(torch.tensor(state, dtype=torch.float, requires_grad=True))
        
        if not terminated:
            q = reward + gamma * torch.max(target_model(torch.tensor(next_state, dtype=torch.float, requires_grad=True)))
        else:
            q = reward

        target = prediction.clone()
        target[action] = q

        optimizer.zero_grad()
        loss = criterion(prediction, target.detach())
        loss.backward()
        optimizer.step()

        return loss

class SARSATrainer:
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
        gamma: float,
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

                # SARSA Update
                next_action = self.epsilon_greedy_policy(q_table, next_state, epsilon)

                q_table[state][action] += learning_rate * (
                    reward + gamma * q_table[next_state][next_action] - q_table[state][action]
                )

                state = next_state

                if terminated or step == max_steps:
                    break

                step += 1

        return q_table
