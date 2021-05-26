from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from typing import List, Dict
from gym.spaces import Space


class Agent(ABC):

    def __init__(self, action_space: Space, obs_space: Space, gamma: float, epsilon: float):

        self.action_space = action_space
        self.obs_space = obs_space
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_table = np.zeros((obs_space.n, action_space.n))

    def act(self, obs: int):

        if np.random.uniform(0, 1) <= self.epsilon:
            a = self.action_space.sample()
        else:
            a = self.q_table[obs, :].argmax()
        return a


class QLearningAgent(Agent):

    def __init__(self, action_space: Space, obs_space: Space, alpha: int, gamma: float, epsilon: float):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(action_space, obs_space, gamma, epsilon)
        self.alpha: float = alpha

        self.epsilon_tracker = []
        self.alpha_tracker = []

    def learn(self, obs, action, n_obs, reward, done) -> float:
        current_q = self.q_table[obs, action]
        max_q_next = self.q_table[n_obs, :].max()
        updated_q = current_q + self.alpha*(reward + (1-done)*self.gamma*max_q_next - current_q)
        self.q_table[obs, action] = updated_q
        return self.q_table[obs, action]

    def schedule_hyperparameters(self, eps_num, total_eps):
        # schedule exploration
        if self.epsilon > 0.15:
            self.epsilon -= 1/total_eps
        self.epsilon_tracker.append(self.epsilon)

        # schedule learning rate
        if self.alpha > 0.15:
            self.alpha -= 1/total_eps
        self.alpha_tracker.append(self.alpha)


class SarsaAgent(Agent):

    def __init__(self, action_space: Space, obs_space: Space, alpha: int, gamma: float, epsilon: float):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(action_space, obs_space, gamma, epsilon)
        self.alpha: float = alpha

        self.epsilon_tracker = []
        self.alpha_tracker = []

    def learn(self, obs, action, n_obs, reward, done) -> float:
        current_q = self.q_table[obs, action]
        n_action = self.act(n_obs)
        next_q = self.q_table[n_obs, n_action]
        updated_q = current_q + self.alpha*(reward + (1-done)*self.gamma*next_q - current_q)
        self.q_table[obs, action] = updated_q
        return self.q_table[obs, action]

    def schedule_hyperparameters(self, eps_num, total_eps):
        # schedule exploration
        if self.epsilon > 0.1:
            self.epsilon -= 1/total_eps
        self.epsilon_tracker.append(self.epsilon)

        # schedule learning rate
        if self.alpha > 0.1:
            self.alpha -= 1/total_eps
        self.alpha_tracker.append(self.alpha)


class MonteCarloAgent(Agent):
    """Agent using the Monte-Carlo algorithm for training
    """

    def __init__(self, action_space: Space, obs_space: Space, gamma: float, epsilon: float):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(action_space, obs_space, gamma, epsilon)
        self.sa_counts = np.zeros((obs_space.n, action_space.n))

    def learn(self, obses: List[np.ndarray], actions: List[int], rewards: List[float]) -> Dict:
        """Updates the Q-table based on agent experience

        :param obses:
            list of received observations representing environmental states of trajectory (in
            the order they were encountered)
        :param actions: list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards: list of received rewards during trajectory (in the order
            they were received)
        """

        G = 0
        T = len(obses)
        sa_pairs = list(zip(obses, actions))
        for t in list(range(T))[::-1]:
            G = self.gamma*G + rewards[t]
            St, At = obses[t], actions[t]
            if (St, At) not in sa_pairs[:t]:
                self.sa_counts[St, At] += 1
                self.q_table[St, At] += 1/self.sa_counts[St, At] * (G - self.q_table[St, At])

    def schedule_hyperparameters(self, eps_num, total_eps):
        # schedule exploration
        if self.epsilon > 0.05:
            self.epsilon -= 1/total_eps
        # self.epsilon_tracker.append(self.epsilon)