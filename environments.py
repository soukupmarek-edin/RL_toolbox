import gym
import numpy as np


class AuctionEnvV1(gym.Env):
    """
    Discrete environment to play with
    """

    def __init__(self, dist, maxbudget, n_bidders, maxsteps, action_granularity=1., randomize_maxbudget=True):
        """
        :param dist: distribution of bids by other bidders
        :param maxbudget: maximum budget of our agent
        :param n_bidders: the number of other bidders in the auction
        :param maxsteps: maximum steps in one episode
        """
        super().__init__()
        self.dist = dist
        self.maxbudget = maxbudget
        self.budget = maxbudget
        self.n_bidders = n_bidders
        self.maxsteps = maxsteps
        self.s = action_granularity
        self.randomize_maxbudget = randomize_maxbudget

        self.actions = np.arange(0, self.maxbudget/4 + self.s, self.s)
        self.budgets = np.arange(0, maxbudget+self.s, self.s)

        self.action_space = gym.spaces.Discrete(self.actions.size)
        self.observation_space = gym.spaces.Discrete(self.budgets.size)

        self.budget_starts = np.arange(self.maxbudget * 0.5, self.maxbudget + self.s, self.s)
        self.budget_start_probs = np.linspace(np.min(self.budget_starts), np.max(self.budget_starts), self.budget_starts.size)[::-1]
        self.budget_start_probs /= self.budget_start_probs.sum()

        self.step_counter = 0

    def step(self, action_id):
        bids = self.dist.sample(self.n_bidders).round(1)
        action = min(self.actions[action_id], self.budget)
        # action = self.actions[action_id]
        # if action > self.budget:
        #     reward = -2
        # else:
        if action >= np.max(bids):
            reward = 1
            self.budget -= action
        else:
            reward = 0

        obs = int(self.budget*1/self.s)

        if self.budget <= 1. or self.step_counter >= self.maxsteps:
            done = True
        else:
            done = False

        self.step_counter += 1

        return obs, reward, done, {}

    def reset(self):
        if self.randomize_maxbudget:
            self.budget = np.random.choice(self.budget_starts, p=self.budget_start_probs)
        else:
            self.budget = self.maxbudget

        obs = int(self.budget*1/self.s)
        self.step_counter = 0
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class AuctionEnvV0(gym.Env):
    """
    Discrete environment without reward shaping
    """

    def __init__(self, dist, maxbudget, n_bidders, maxsteps):
        """
        :param dist: distribution of bidders. Instance of scipy.stats distribution
        :param maxbudget: maximum budget of our agent
        :param n_bidders: the number of other bidders in the auction
        :param maxsteps: maximum steps in one episode
        """
        super().__init__()
        self.dist = dist

        self.action_space = gym.spaces.Discrete(maxbudget + 1)
        self.observation_space = gym.spaces.Discrete(maxbudget + 1)
        self.maxbudget = maxbudget
        self.obs = maxbudget
        self.n_bidders = n_bidders
        self.maxsteps = maxsteps

        self.step_counter = 0

    def step(self, action):
        bids = self.dist.rvs(self.n_bidders)
        action = min(action, self.obs)

        if action >= np.max(bids):
            reward = 1
            self.obs -= action
        else:
            reward = 0

        if self.obs <= 0 or self.step_counter == self.maxsteps:
            done = True
            reward = -self.obs
        else:
            done = False

        self.step_counter += 1

        return self.obs, reward, done, {}

    def reset(self):
        self.obs = self.maxbudget
        self.step_counter = 0
        return self.obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass
