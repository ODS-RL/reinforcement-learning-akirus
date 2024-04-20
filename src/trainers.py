import numpy as np
import torch
import random
from collections import deque
from tqdm import tqdm
from src.envs.base_env import BaseEnvironment

# https://arxiv.org/pdf/1312.5602.pdf
# https://arxiv.org/pdf/1509.02971.pdf

class BaseTrainer:
    def __init__(self, env: BaseEnvironment) -> None:
        self.env = env
        self.epsilon_decays = {
            "exponential": self.exponential_decay,
            "linear": self.linear_decay
        }

    def greedy_policy(self, q_table: np.ndarray, state: tuple[int, int]) -> int:
        return np.argmax(q_table[state])

    def epsilon_greedy_policy(self,
        q_table: np.ndarray, state: tuple[int, int], epsilon: float
    ) -> int:
        """Epsilon-greedy Action Selection"""
        if np.random.uniform(0, 1) < epsilon:
            # return np.random.choice(len(self.env.actions))
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(q_table, state)
        
    def exponential_decay(self, step, epsilon_start: float, epsilon_end: float, n_steps: int, decay_rate: int = 1) -> float:
        assert 1 >= epsilon_start >= epsilon_end >= 0
        assert 1 >= decay_rate >= 0

        if decay_rate == 0:
            return self.linear_decay(step, epsilon_start, epsilon_end, n_steps, decay_rate)

        magnitude = n_steps
        # https://www.desmos.com/calculator/ta2owwskqm
        # https://math.stackexchange.com/questions/4014286/exponential-between-2-points
        # return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step_idx * decay_rate)
        return (epsilon_end - epsilon_start) * (np.exp(-decay_rate * magnitude * step/n_steps) - 1) / (np.exp(-decay_rate * magnitude) - 1) + epsilon_start

    def linear_decay(self, step: int, epsilon_start: float, epsilon_end: float, n_steps: int, decay_rate: int = 1):
        assert 1 >= epsilon_start >= epsilon_end >= 0
        assert 1 >= decay_rate >= 0

        if decay_rate == 1:
            return epsilon_end
    
        fraction = min(float(step) / int((1 - decay_rate) * n_steps), 1.0)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)
        
    def train(
        self
    ) -> np.ndarray:
        raise NotImplementedError
    


class QLearningTrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(env)

    def train(
        self,
        q_table: np.ndarray,
        n_episodes: int,
        learning_rate: float,
        gamma: float,
        max_steps: int = None,
        epsilon: float = None,
        max_epsilon: float = None,
        min_epsilon: float = None,
        decay_rate: float = None,
        decay_epsilon: str = "constant",
    ) -> np.ndarray:
        for episode in tqdm(range(n_episodes), total=n_episodes):
            state, _ = self.env.reset()

            epsilon = self.epsilon_decays[decay_epsilon](episode, max_epsilon, min_epsilon, n_episodes, decay_rate) if decay_epsilon != "constant" else epsilon

            step = 0
            while True:
                action = self.epsilon_greedy_policy(q_table, state, epsilon)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # The Bellman Equation
                q_table[state][action] += learning_rate * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
                )

                state = next_state

                if done or step == max_steps:
                    break

                step += 1

        return q_table



class ReplayMemory(object):
    def __init__(self, memory_size: int, batch_size: int) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=memory_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class DQNTrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment, memory: ReplayMemory = None, device: str = None) -> None:
        super().__init__(env)
        self.memory = memory
        self.device = device

    def greedy_policy(self, model, state: tuple[int, int]) -> int:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return torch.argmax(model(state))    
        
    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        n_episodes: int,
        gamma: float,
        max_steps: int = None,
        epsilon: float = None,
        max_epsilon: float = None,
        min_epsilon: float = None,
        decay_rate: float = None,
        decay_epsilon: str = "constant",
    ) -> np.ndarray:
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        scores = []
        scores_window = deque(maxlen = 100)
        for episode in tqdm_range:
            state, _ = self.env.reset()

            epsilon = self.epsilon_decays[decay_epsilon](episode, max_epsilon, min_epsilon, n_episodes, decay_rate) if decay_epsilon != "constant" else epsilon

            step = 0
            score = 0
            while True:
                action = self.epsilon_greedy_policy(model, state, epsilon)
                
                next_state, reward, terminated, truncated, _  = self.env.step(action.item())

                done = terminated or truncated
                score += reward

                if self.memory is not None:
                    self.memory.add(state, action, reward, next_state, done)

                    if len(self.memory) >= self.memory.batch_size:
                        transitions = self.memory.sample()

                        # for transition in transitions:
                        #     self._train(model, optimizer, criterion, gamma, transition)

                        self._batch_train(model, optimizer, criterion, gamma, transitions)
                else:
                    self._train(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        gamma=gamma,
                        transition=(state, action, reward, next_state, done),
                    )

                state = next_state

                if done or step == max_steps:
                    break

                step += 1
                
            scores.append(score)
            scores_window.append(score)   
            tqdm_range.set_description(f"Score: {np.mean(scores_window):.3f}")

        return scores
    
    def _train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transition,
    ):
        state, action, reward, next_state, done = transition
        prediction = model(torch.tensor(state, dtype=torch.float).to(self.device))
        target = prediction.clone()
        
        if not done:
            q = reward + gamma * torch.max(model(torch.tensor(next_state, dtype=torch.float).to(self.device)))
        else:
            q = reward

        target[action] = q

        optimizer.zero_grad()
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        return loss
    
    def _batch_train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transitions
    ):
        states, actions, rewards, next_states, dones = map(list, zip(*transitions))
        
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        prediction = model(states_tensor)
        
        target = prediction.clone()
        
        next_state_values = model(next_states_tensor)

        max_next_state_values = torch.max(next_state_values, dim=1)[0]

        q_values = rewards_tensor + gamma * max_next_state_values * (~dones_tensor)
        
        target[range(len(actions)), actions] = q_values
        
        optimizer.zero_grad()
        loss = criterion(prediction, target.detach())
        loss.backward()
        optimizer.step()


class DDQNTrainer(BaseTrainer):
    """Double DQN Trainer"""
    # https://arxiv.org/pdf/1509.06461.pdf
    def __init__(self, env: BaseEnvironment, memory: ReplayMemory = None, device = None) -> None:
        super().__init__(env)
        self.memory = memory
        self.device = device

    def greedy_policy(self, model, state: tuple[int, int]) -> int:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return torch.argmax(model(state))    
                
    def train(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        n_episodes: int,
        gamma: float,
        tau = 0.005,
        max_steps: int = None,
        epsilon: float = None,
        max_epsilon: float = None,
        min_epsilon: float = None,
        decay_rate: float = None,
        decay_epsilon: str = "constant",
    ) -> np.ndarray:
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        scores = []
        scores_window = deque(maxlen = 100)
        for episode in tqdm_range:
            state, _ = self.env.reset()

            epsilon = self.epsilon_decays[decay_epsilon](episode, max_epsilon, min_epsilon, n_episodes, decay_rate) if decay_epsilon != "constant" else epsilon

            score = 0
            step = 0
            while True:
                action: int = self.epsilon_greedy_policy(online_model, state, epsilon)

                next_state, reward, terminated, truncated, _  = self.env.step(action.item())

                done = terminated or truncated
                score += reward

                if self.memory is not None:
                    self.memory.add( state, action, reward, next_state, done)

                    if len(self.memory) >= self.memory.batch_size:
                        transitions = self.memory.sample()

                        # for transition in transitions:
                        #     self._train(online_model, target_model, optimizer, criterion, gamma, transition)

                        self._batch_train(
                            online_model=online_model,
                            target_model=target_model,
                            optimizer=optimizer,
                            criterion=criterion,
                            gamma=gamma,
                            transitions=transitions
                        )

                else:
                    self._train(
                        online_model=online_model,
                        target_model=target_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        gamma=gamma,
                        transition=(state, action, reward, next_state, done),
                    )

                target_weights = target_model.state_dict()
                policy_weights = online_model.state_dict()
                for key in policy_weights:
                    target_weights[key] = policy_weights[key]*tau + target_weights[key]*(1-tau)
                target_model.load_state_dict(target_weights)

                state = next_state

                if done or step == max_steps:
                    break

                step += 1
            
            scores.append(score)
            scores_window.append(score)   
            tqdm_range.set_description(f"Score: {np.mean(scores_window):.3f}")

        return scores
    
    def _train(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transition,
    ):
        state, action, reward, next_state, done = transition
        prediction = online_model(torch.tensor(state, dtype=torch.float).to(self.device))
        
        if not done:
            q = reward + gamma * torch.max(target_model(torch.tensor(next_state, dtype=torch.float).to(self.device)))
        else:
            q = reward

        target = prediction.clone()
        target[action] = q

        optimizer.zero_grad()
        loss = criterion(prediction, target.detach())
        loss.backward()
        optimizer.step()

    
    def _batch_train(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transitions,
    ):
        states, actions, rewards, next_states, dones = map(list, zip(*transitions))
        
        states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)

        predictions = online_model(states_tensor)

        next_state_values = target_model(next_states_tensor)

        max_next_state_values = torch.max(next_state_values, dim=1)[0]

        q_values = rewards_tensor + (1 - dones_tensor.float()) * gamma * max_next_state_values

        targets = predictions.clone()

        targets[range(len(actions)), actions] = q_values

        optimizer.zero_grad()
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()


class SARSATrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(env)

    def train(
        self,
        q_table: np.ndarray,
        n_episodes: int,
        learning_rate: float,
        gamma: float,
        max_steps: int = None,
        epsilon: float = None,
        max_epsilon: float = None,
        min_epsilon: float = None,
        decay_rate: float = None,
        decay_epsilon: str = "constant",
    ) -> np.ndarray:
        for episode in tqdm(range(n_episodes), total=n_episodes):
            state, _ = self.env.reset()

            epsilon = self.epsilon_decays[decay_epsilon](episode, max_epsilon, min_epsilon, n_episodes, decay_rate) if decay_epsilon != "constant" else epsilon

            step = 0
            while True:
                action = self.epsilon_greedy_policy(q_table, state, epsilon)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # SARSA Update
                next_action = self.epsilon_greedy_policy(q_table, next_state, epsilon)

                q_table[state][action] += learning_rate * (
                    reward + gamma * q_table[next_state][next_action] - q_table[state][action]
                )

                state = next_state

                if done or step == max_steps:
                    break

                step += 1

        return q_table
