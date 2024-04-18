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

class DQNMemoryTrainer:
    "Modification of the DQN with memory replay algorithm"
    def __init__(self, env: BaseEnvironment, mem_size: int, batch_size: int) -> None:
        self.env = env
        self.memory_size = mem_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=mem_size)

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

            # Not original implementation
            # while True:
            #     action = self.epsilon_greedy_policy(model, state, epsilon)

            #     next_state, reward, terminated = self.env.step(action)

            #     self.memory.append((state, action, next_state, reward, terminated))

            #     loss = self._train(model, optimizer, criterion, gamma, (state, action, next_state, reward, terminated))

            #     episode_losses.append(loss.item())

            #     state = next_state

            #     if terminated or step == max_steps:
            #         if len(self.memory) > self.batch_size:
            #             transitions = random.sample(self.memory, self.batch_size)

            #             for transition in transitions:
            #                 loss = self._train(model, optimizer, criterion, gamma, transition)

            #                 episode_losses.append(loss.item())

            #         break

            #     step += 1

            while True:
                action = self.epsilon_greedy_policy(model, state, epsilon)

                next_state, reward, terminated = self.env.step(action)

                self.memory.append((state, action, next_state, reward, terminated))

                state = next_state

                if len(self.memory) > self.batch_size:
                    transitions = random.sample(self.memory, self.batch_size)

                    for transition in transitions:
                        loss = self._train(model, optimizer, criterion, gamma, transition)

                        episode_losses.append(loss.item())

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
    def __init__(self, env: BaseEnvironment, mem_size: int, batch_size: int) -> None:
        self.env = env
        self.memory_size = mem_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=mem_size)

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
        gamma,
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

                self.memory.append((state, action, next_state, reward, terminated))

                state = next_state

                if len(self.memory) > self.batch_size:
                    transitions = random.sample(self.memory, self.batch_size)

                    for transition in transitions:
                        loss = self._train(online_model, target_model, optimizer, criterion, gamma, transition)

                        episode_losses.append(loss.item())

                target_weights = target_model.state_dict()
                policy_weights = online_model.state_dict()
                for key in policy_weights:
                    target_weights[key] = policy_weights[key]*tau + target_weights[key]*(1-tau)
                target_model.load_state_dict(target_weights)

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

class SarsaTrainer: # TODO: Implement this
    def __init__(self):
        pass 


