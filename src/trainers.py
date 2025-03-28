import numpy as np
import torch
from torch.distributions.categorical import Categorical
from collections import deque
from tqdm import tqdm
from src.envs.base_env import BaseEnvironment
from src.noise import BaseNoise
from src.buffers import ReplayMemory, Buffer, ExtendableBuffer

# https://arxiv.org/pdf/1312.5602.pdf
# https://arxiv.org/pdf/1509.02971.pdf

# TODO: Split BaseTrainer into BaseQLearningTrainer and BasePolicyOptimizationTrainer


class BaseTrainer:
    def __init__(self, env: BaseEnvironment) -> None:
        self.env = env
        self.epsilon_decays = {
            "exponential": self.exponential_decay,
            "linear": self.linear_decay,
        }

    def greedy_policy(self, q_table: np.ndarray, state: tuple[int, int]) -> int:
        return np.argmax(q_table[state])

    def epsilon_greedy_policy(
        self, q_table: np.ndarray, state: tuple[int, int], epsilon: float
    ) -> int:
        """Epsilon-greedy Action Selection"""
        if np.random.uniform(0, 1) < epsilon:
            # return np.random.choice(len(self.env.actions))
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(q_table, state)

    def exponential_decay(
        self,
        step,
        epsilon_start: float,
        epsilon_end: float,
        n_steps: int,
        decay_rate: int = 1,
    ) -> float:
        assert 1 >= epsilon_start >= epsilon_end >= 0
        assert 1 >= decay_rate >= 0

        if decay_rate == 0:
            return self.linear_decay(
                step, epsilon_start, epsilon_end, n_steps, decay_rate
            )

        magnitude = n_steps
        # https://www.desmos.com/calculator/ta2owwskqm
        # https://math.stackexchange.com/questions/4014286/exponential-between-2-points
        # return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step_idx * decay_rate)
        return (epsilon_end - epsilon_start) * (
            np.exp(-decay_rate * magnitude * step / n_steps) - 1
        ) / (np.exp(-decay_rate * magnitude) - 1) + epsilon_start

    def linear_decay(
        self,
        step: int,
        epsilon_start: float,
        epsilon_end: float,
        n_steps: int,
        decay_rate: int = 1,
    ):
        assert 1 >= epsilon_start >= epsilon_end >= 0
        assert 1 >= decay_rate >= 0

        if decay_rate == 1:
            return epsilon_end

        fraction = min(float(step) / int((1 - decay_rate) * n_steps), 1.0)
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    def train(self) -> np.ndarray:
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

            epsilon = (
                self.epsilon_decays[decay_epsilon](
                    episode, max_epsilon, min_epsilon, n_episodes, decay_rate
                )
                if decay_epsilon != "constant"
                else epsilon
            )

            step = 0
            while True:
                action = self.epsilon_greedy_policy(q_table, state, epsilon)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # The Bellman Equation
                q_table[state][action] += learning_rate * (
                    reward
                    + gamma * np.max(q_table[next_state])
                    - q_table[state][action]
                )

                state = next_state

                if done or step == max_steps:
                    break

                step += 1

        return q_table


class DQNTrainer(BaseTrainer):
    def __init__(
        self, env: BaseEnvironment, memory: ReplayMemory = None, device: str = None
    ) -> None:
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
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()

            epsilon = (
                self.epsilon_decays[decay_epsilon](
                    episode, max_epsilon, min_epsilon, n_episodes, decay_rate
                )
                if decay_epsilon != "constant"
                else epsilon
            )

            step = 0
            score = 0
            while True:
                action = self.epsilon_greedy_policy(model, state, epsilon)

                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                done = terminated or truncated
                score += reward

                if self.memory is not None:
                    self.memory.add(state, action, reward, next_state, done)

                    if len(self.memory) >= self.memory.batch_size:
                        transitions = self.memory.sample()

                        # for transition in transitions:
                        #     self._train(model, optimizer, criterion, gamma, transition)

                        self._batch_train(
                            model, optimizer, criterion, gamma, transitions
                        )
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
            q = reward + gamma * torch.max(
                model(torch.tensor(next_state, dtype=torch.float).to(self.device))
            )
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
        transitions,
    ):
        states, actions, rewards, next_states, dones = map(list, zip(*transitions))

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device
        )
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(
            self.device
        )
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

    # https://arxiv.org/pdf/1509.06461.pdf (Deep Reinforcement Learning with Double Q-learning)
    def __init__(
        self, env: BaseEnvironment, memory: ReplayMemory = None, device=None
    ) -> None:
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
        tau=0.005,
        max_steps: int = None,
        epsilon: float = None,
        max_epsilon: float = None,
        min_epsilon: float = None,
        decay_rate: float = None,
        decay_epsilon: str = "constant",
        n_steps_update: int = 1,
    ) -> np.ndarray:
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        scores = []
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()

            epsilon = (
                self.epsilon_decays[decay_epsilon](
                    episode, max_epsilon, min_epsilon, n_episodes, decay_rate
                )
                if decay_epsilon != "constant"
                else epsilon
            )

            score = 0
            step = 0
            while True:
                action: int = self.epsilon_greedy_policy(online_model, state, epsilon)

                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                done = terminated or truncated
                score += reward

                if self.memory is not None:
                    self.memory.add(state, action, reward, next_state, done)

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
                            transitions=transitions,
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

                if step % n_steps_update == 0:
                    self._soft_update(target_model, online_model, tau)

                state = next_state

                if done or step == max_steps:
                    break

                step += 1

            scores.append(score)
            scores_window.append(score)
            tqdm_range.set_description(f"Score: {np.mean(scores_window):.3f}")

        return scores

    def _soft_update(
        self, target: torch.nn.Module, source: torch.nn.Module, tau: float
    ):
        target_weights = target.state_dict()
        source_weights = source.state_dict()
        for key in source_weights:
            target_weights[key] = source_weights[key] * tau + target_weights[key] * (
                1 - tau
            )
        target.load_state_dict(target_weights)

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
        prediction = online_model(
            torch.tensor(state, dtype=torch.float).to(self.device)
        )

        if not done:
            q = reward + gamma * torch.max(
                target_model(
                    torch.tensor(next_state, dtype=torch.float).to(self.device)
                )
            )
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

        states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(
            self.device
        )
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(
            self.device
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)

        predictions = online_model(states_tensor)

        next_state_values = target_model(next_states_tensor)

        max_next_state_values = torch.max(next_state_values, dim=1)[0]

        q_values = (
            rewards_tensor + (1 - dones_tensor.float()) * gamma * max_next_state_values
        )

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

            epsilon = (
                self.epsilon_decays[decay_epsilon](
                    episode, max_epsilon, min_epsilon, n_episodes, decay_rate
                )
                if decay_epsilon != "constant"
                else epsilon
            )
            action = self.epsilon_greedy_policy(q_table, state, epsilon)

            step = 0
            while True:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # SARSA Update
                next_action = self.epsilon_greedy_policy(q_table, next_state, epsilon)

                q_table[state][action] += learning_rate * (
                    reward
                    + gamma * q_table[next_state][next_action]
                    - q_table[state][action]
                )

                state = next_state
                action = next_action

                if done or step == max_steps:
                    break

                step += 1

        return q_table


class DDPGTrainer(BaseTrainer):
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def __init__(
        self, env: BaseEnvironment, memory: ReplayMemory = None, device=None
    ) -> None:
        super().__init__(env)
        self.memory = memory
        self.device = device
        assert self.memory is not None, "Memory cannot be None"

    def apply_noise(self, action, noise):
        lower_bound = self.env.action_space.low.item()
        upper_bound = self.env.action_space.high.item()
        return np.clip(action + noise, lower_bound, upper_bound)

    def policy(self, actor, state):
        return (
            actor(torch.tensor(state, dtype=torch.float).to(self.device))
            .cpu()
            .detach()
            .numpy()
        )

    def train(
        self,
        actor,
        critic,
        actor_target,
        critic_target,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        noise: BaseNoise,
        n_episodes: int,
        gamma: float,
        tau=0.005,
        max_steps: int = None,
    ) -> np.ndarray:
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        scores = []
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()
            noise.reset()

            score = 0
            step = 0
            while True:
                action = self.policy(actor, state)

                action = self.apply_noise(action, noise.sample())

                next_state, reward, terminated, truncated, _ = self.env.step(
                    [action.item()]
                )

                done = terminated or truncated
                score += reward

                self.memory.add(state, action, [reward], next_state, [done])

                if len(self.memory) >= self.memory.batch_size:
                    transitions = self.memory.sample()

                    self._batch_train(
                        actor=actor,
                        critic=critic,
                        actor_target=actor_target,
                        critic_target=critic_target,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        criterion=criterion,
                        gamma=gamma,
                        transitions=transitions,
                    )

                self._soft_update(actor_target, actor, tau)
                self._soft_update(critic_target, critic, tau)

                state = next_state

                if done or step == max_steps:
                    break

                step += 1

            scores.append(score)
            scores_window.append(score)
            tqdm_range.set_description(f"Score: {np.mean(scores_window):.3f}")

        return scores

    def _batch_train(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        actor_target: torch.nn.Module,
        critic_target: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        gamma,
        transitions,
    ):
        states, actions, rewards, next_states, dones = map(list, zip(*transitions))

        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(
            self.device
        )
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        critic_values = critic(states, actions)

        next_actions = actor_target(next_states)
        next_q_values = critic_target(next_states, next_actions.detach())
        y = rewards + gamma * (1 - dones) * next_q_values

        critic_loss = criterion(y, critic_values)

        policy_loss = -critic(states, actor(states)).mean()

        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    def _soft_update(
        self, target: torch.nn.Module, source: torch.nn.Module, tau: float
    ):
        target_weights = target.state_dict()
        source_weights = source.state_dict()
        for key in source_weights:
            target_weights[key] = source_weights[key] * tau + target_weights[key] * (
                1 - tau
            )
        target.load_state_dict(target_weights)


class VPGTrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment, device=None) -> None:
        super().__init__(env)
        self.device = device
        self.buffer = Buffer()
        self.batch_buffer = ExtendableBuffer()

    def policy(self, actor, state):
        probs = actor(torch.tensor(state, dtype=torch.float).to(self.device))
        dist = Categorical(probs)
        action = dist.sample()
        return action.cpu().detach().numpy()

    def reward_to_go(self, rewards, dones, gamma=1.0, noralize=False):
        # https://subscription.packtpub.com/book/data/9781789533583/1/ch01lvl1sec05/identifying-reward-functions-and-the-concept-of-discounted-rewards
        # https://medium.com/iecse-hashtag/rl-part-2-returns-policy-and-value-functions-33311f16197
        rewards_to_go = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            cumulative_reward = reward + gamma * (1 - done) * cumulative_reward
            rewards_to_go.insert(0, cumulative_reward)

        # for i in range(len(rewards)):
        #     for j in range(i, len(rewards)):
        #         rewards_to_go[i] += rewards[j] * (1 - dones[j]) * (gamma ** (j - i))

        # rewards_to_go = np.array([np.sum(rewards[i:]*(1 - dones[i:])*(gamma**np.array(range(0, len(rewards)-i)))) for i in range(len(rewards))])

        if noralize:
            rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (
                np.std(rewards_to_go) + 1e-9
            )

        return rewards_to_go

    def train(
        self,
        actor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_episodes: int,
        batch_size: int,
        gamma: float = 0.99,
        max_steps: int = None,
    ):
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        self.batch_buffer.reset()
        scores = []
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()
            self.buffer.reset()

            score = 0
            step = 0
            while True:
                action = self.policy(actor, state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                done = terminated or truncated

                self.buffer.add(state, action, reward, done)

                score += reward

                state = next_state

                if done or step == max_steps:
                    cache = self.buffer.take()
                    states, actions, rewards, dones = map(np.array, zip(*cache))

                    rewards_to_go = self.reward_to_go(rewards, dones, gamma)

                    self.batch_buffer.extend(states, actions, rewards_to_go)

                    if len(self.batch_buffer) >= batch_size:
                        batch_cache = self.batch_buffer.take()

                        batch_states, batch_actions, batch_rewards_to_go = batch_cache

                        batch_states = torch.tensor(
                            np.array(batch_states), dtype=torch.float
                        ).to(self.device)
                        batch_actions = torch.tensor(
                            np.array(batch_actions), dtype=torch.int64
                        ).to(self.device)
                        batch_rewards_to_go = torch.tensor(
                            np.array(batch_rewards_to_go), dtype=torch.float
                        ).to(self.device)

                        logprobs = torch.log(actor(batch_states))
                        actions_logprobs = (
                            batch_rewards_to_go
                            * torch.gather(
                                logprobs, 1, batch_actions.unsqueeze(1)
                            ).squeeze()
                        )  # or logprob[torch.arange(len(logprob)), batch_actions].squeeze()

                        loss = -actions_logprobs.mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        self.batch_buffer.reset()

                    break

                step += 1

            scores.append(score)
            scores_window.append(score)
            tqdm_range.set_description(f"Score: {np.mean(scores_window):.3f}")

        return scores


class A2CTrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment, device=None) -> None:
        super().__init__(env)
        self.device = device
        self.buffer = Buffer()
        self.batch_buffer = ExtendableBuffer()

    def policy(self, actor, critic, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        probs = actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        value = critic(state)

        entropy = dist.entropy().mean()

        return action.cpu().detach().numpy(), value.cpu().detach().numpy()

    def reward_to_go(self, rewards, dones, gamma=1.0, noralize=False):
        # https://subscription.packtpub.com/book/data/9781789533583/1/ch01lvl1sec05/identifying-reward-functions-and-the-concept-of-discounted-rewards
        # https://medium.com/iecse-hashtag/rl-part-2-returns-policy-and-value-functions-33311f16197
        rewards_to_go = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            cumulative_reward = reward + gamma * (1 - done) * cumulative_reward
            rewards_to_go.insert(0, cumulative_reward)

        if noralize:
            rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (
                np.std(rewards_to_go) + 1e-9
            )

        return rewards_to_go

    def train(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        n_episodes: int,
        batch_size: int,
        gamma: float = 0.99,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_steps: int = None,
    ):
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        self.batch_buffer.reset()
        scores = []
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()
            self.buffer.reset()

            score = 0
            step = 0
            while True:
                action, value = self.policy(actor, critic, state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                done = terminated or truncated

                self.buffer.add(state, action, value, reward, done)

                score += reward

                state = next_state

                if done or step == max_steps:
                    cache = self.buffer.take()
                    states, actions, values, rewards, dones = map(np.array, zip(*cache))

                    rewards_to_go = self.reward_to_go(rewards, dones, gamma)

                    self.batch_buffer.extend(states, actions, values, rewards_to_go)

                    if len(self.batch_buffer) >= batch_size:
                        batch_cache = self.batch_buffer.take()

                        (
                            batch_states,
                            batch_actions,
                            batch_values,
                            batch_rewards_to_go,
                        ) = batch_cache

                        batch_states = torch.tensor(
                            np.array(batch_states), dtype=torch.float
                        ).to(self.device)
                        batch_actions = torch.tensor(
                            np.array(batch_actions), dtype=torch.int64
                        ).to(self.device)
                        batch_values = torch.tensor(
                            np.array(batch_values), dtype=torch.float
                        ).to(self.device)
                        batch_rewards_to_go = torch.tensor(
                            np.array(batch_rewards_to_go), dtype=torch.float
                        ).to(self.device)

                        probs = actor(batch_states)
                        dist = Categorical(probs)
                        entropy = dist.entropy().mean()

                        advantages = batch_rewards_to_go - batch_values.squeeze()

                        logprobs = torch.log(actor(batch_states))
                        actions_logprobs = (
                            advantages
                            * torch.gather(
                                logprobs, 1, batch_actions.unsqueeze(1)
                            ).squeeze()
                        )  # or logprob[torch.arange(len(logprob)), batch_actions].squeeze()
                        actor_loss = -actions_logprobs.mean()

                        critic_loss = criterion(
                            critic(batch_states).squeeze(), batch_rewards_to_go
                        )

                        loss = (
                            actor_loss
                            + value_coeff * critic_loss
                            - entropy_coeff * entropy
                        )

                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()

                        self.batch_buffer.reset()

                    break

                step += 1

            scores_window.append(score)
            scores.append(score)
            tqdm_range.set_description(f"Score: {np.mean(scores_window)}")

        return scores


class PPOTrainer(BaseTrainer):
    def __init__(self, env: BaseEnvironment, device=None) -> None:
        super().__init__(env)
        self.device = device
        self.buffer = Buffer()
        self.batch_buffer = ExtendableBuffer()

    def policy(self, actor, critic, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        probs = actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        value = critic(state)

        return (
            action.cpu().detach().numpy(),
            value.cpu().detach().numpy(),
            dist.log_prob(action).cpu().detach().numpy(),
        )  # The same as torch.log(probs)

    def reward_to_go(self, rewards, dones, gamma=1.0, normalize=False):
        rewards_to_go = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            cumulative_reward = reward + gamma * (1 - done) * cumulative_reward
            rewards_to_go.insert(0, cumulative_reward)

        if normalize:
            rewards_to_go = (rewards_to_go - np.mean(rewards_to_go)) / (
                np.std(rewards_to_go) + 1e-9
            )

        return rewards_to_go

    def train(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        n_episodes: int,
        batch_size: int,
        ppo_epochs: int = 10,
        gamma: float = 0.99,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        clip_param: float = 0.2,
        max_steps: int = None,
    ):
        tqdm_range = tqdm(range(n_episodes), total=n_episodes)

        self.batch_buffer.reset()
        scores = []
        scores_window = deque(maxlen=100)
        for episode in tqdm_range:
            state, _ = self.env.reset()
            self.buffer.reset()

            score = 0
            step = 0
            while True:
                action, value, log_prob = self.policy(actor, critic, state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                done = terminated or truncated

                self.buffer.add(state, action, value, reward, done, log_prob)

                score += reward

                state = next_state

                if done or step == max_steps:
                    cache = self.buffer.take()
                    states, actions, values, rewards, dones, log_probs = map(
                        np.array, zip(*cache)
                    )

                    rewards_to_go = self.reward_to_go(rewards, dones, gamma)

                    self.batch_buffer.extend(
                        states, actions, values, rewards_to_go, log_probs
                    )

                    if len(self.batch_buffer) >= batch_size:
                        batch_cache = self.batch_buffer.take()

                        (
                            batch_states,
                            batch_actions,
                            batch_values,
                            batch_rewards_to_go,
                            batch_old_log_probs,
                        ) = batch_cache

                        batch_states = torch.tensor(
                            np.array(batch_states), dtype=torch.float
                        ).to(self.device)
                        batch_actions = torch.tensor(
                            np.array(batch_actions), dtype=torch.int64
                        ).to(self.device)
                        batch_values = torch.tensor(
                            np.array(batch_values), dtype=torch.float
                        ).to(self.device)
                        batch_rewards_to_go = torch.tensor(
                            np.array(batch_rewards_to_go), dtype=torch.float
                        ).to(self.device)
                        batch_old_log_probs = torch.tensor(
                            np.array(batch_old_log_probs), dtype=torch.float
                        ).to(self.device)

                        advantages = batch_rewards_to_go - batch_values.squeeze()

                        for _ in range(ppo_epochs):
                            probs = actor(batch_states)
                            dist = Categorical(probs)
                            new_log_probs = dist.log_prob(
                                batch_actions
                            )  # The same as torch.log(probs)
                            entropy = dist.entropy().mean()

                            ratio = torch.exp(new_log_probs - batch_old_log_probs)
                            surr1 = ratio * advantages
                            surr2 = (
                                torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                                * advantages
                            )
                            actor_loss = -torch.min(surr1, surr2).mean()

                            critic_loss = criterion(
                                critic(batch_states).squeeze(), batch_rewards_to_go
                            )

                            loss = (
                                actor_loss
                                + value_coeff * critic_loss
                                - entropy_coeff * entropy
                            )

                            actor_optimizer.zero_grad()
                            critic_optimizer.zero_grad()
                            loss.backward()
                            actor_optimizer.step()
                            critic_optimizer.step()

                        self.batch_buffer.reset()

                    break

                step += 1

            scores_window.append(score)
            scores.append(score)
            tqdm_range.set_description(f"Score: {np.mean(scores_window)}")

        return scores
