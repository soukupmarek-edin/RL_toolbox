import numpy as np


class ValueIteration:
    """
    :param transition_matrix (states, actions, states): 3D matrix containing probability of transition.
    :param reward_matrix (states, actions, states): 3D matrix containing rewards in transition
    :param gamma:
    :param theta:
    """
    def __init__(self,
                 transition_matrix: np.ndarray, reward_matrix: np.ndarray,
                 gamma: float = 0.99):

        self.P = transition_matrix
        self.R = reward_matrix
        self.gamma = gamma

        self.states = np.arange(self.P.shape[0])
        self.actions = np.arange(self.P.shape[1])

    def calculate_value_function(self, theta=1e-5):
        """
        :return V (np.ndarray): iterated value function
        """
        states, P, R, gamma = self.states, self.P, self.R, self.gamma
        V = np.zeros(states.size)

        while True:
            diff = 0
            v = np.copy(V)
            for s in range(states.size):
                V[s] = (P[s] * (R[s] + gamma * v)).sum(axis=1).max()
            diff = np.max([diff, np.max(np.abs(V - v))])
            if diff <= theta:
                break
        return V

    def calculate_policy(self, V):
        states, actions, P, R, gamma = self.states, self.actions, self.P, self.R, self.gamma

        policy = np.zeros((states.size, actions.size))

        for s in range(states.size):
            a = (P[s] * (R[s] + gamma * V)).sum(axis=1).argmax()
            policy[s, a] = 1

        return policy

    def solve(self, theta=1e-5):
        """
        :param theta:
        :return: policy, value function
        """
        V = self.calculate_value_function(theta)
        policy = self.calculate_policy(V)
        return policy, V


class PolicyIteration:

    def __init__(self,
                 transition_matrix: np.ndarray, reward_matrix: np.ndarray,
                 gamma: float = 0.99):
        self.P = transition_matrix
        self.R = reward_matrix
        self.gamma = gamma

        self.states = np.arange(self.P.shape[0])
        self.actions = np.arange(self.P.shape[1])

    def evaluate_policy(self, policy, theta=1e-6):
        V = np.zeros(self.states.size)
        P = self.P
        R = self.R
        gamma = self.gamma

        while True:
            diff = 0
            v = np.copy(V)
            for s in range(self.states.size):
                V[s] = np.sum(policy[s] * np.sum(P[s] * (R[s] + gamma * v), axis=1))
            diff = np.max([diff, np.abs(v - V).max()])
            if diff <= theta:
                break
        return V

    def improve_policy(self, theta=1e-6):
        s_dim = self.states.size
        a_dim = self.actions.size
        policy = np.ones((s_dim, a_dim))/np.ones((s_dim, a_dim)).sum(axis=1).reshape(-1, 1)
        P = self.P
        R = self.R
        gamma = self.gamma

        while True:
            V = self.evaluate_policy(policy, theta)
            new_policy = np.zeros([s_dim, a_dim])

            for s in range(s_dim):
                a = np.sum(P[s] * (R[s] + gamma * V), axis=1).argmax()
                new_policy[s, a] = 1

            if np.all(policy == new_policy):
                return policy, V
            else:
                policy = new_policy

    def solve(self, theta=1e-6):
        policy, V = self.improve_policy(theta)
        return policy, V