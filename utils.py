import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate


def evaluate(env, agent, max_steps, eval_episodes):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env: environment to execute evaluation on
    :param agent: agent to act in environment
    :param max_steps: max number of steps per evaluation episode
    :param eval_episodes: number of evaluation episodes
    :return (float, int): mean of returns received over episodes and number of negative
        return evaluation, episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        obs = env.reset()
        episodic_return = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            act = agent.act(obs)
            n_obs, reward, done, info = env.step(act)

            episodic_return += reward
            steps += 1

            obs = n_obs

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    negative_returns = sum([ret < 0 for ret in episodic_returns])

    return mean_return, negative_returns


class Distribution:

    def __init__(self, dist):
        self.dist = dist

    def cdf(self, x):
        return self.dist.cdf(x)

    def fos_cdf(self, x, N):
        cdf = self.dist.cdf
        return cdf(x) ** N

    def sos_cdf(self, x, N):
        cdf = self.dist.cdf
        return N * cdf(x) ** (N - 1) - (N - 1) * cdf(x) ** N

    def exp(self):
        return self.dist.expect()

    def sample(self, N):
        return self.dist.rvs(N)


class ContinuousDistribution(Distribution):

    def __init__(self, dist):
        super().__init__(dist)
        self.dist = dist

    def pdf(self, x):
        return self.dist.pdf(x)

    def fos_pdf(self, x, N):
        cdf, pdf = self.dist.cdf, self.dist.pdf
        return N * cdf(x) ** (N - 1) * pdf(x)

    def sos_pdf(self, x, N):
        cdf, pdf = self.dist.cdf, self.dist.pdf
        return N * (N - 1) * (1 - cdf(x)) * cdf(x) ** (N - 2) * pdf(x)

    def exp_fos(self, N):
        return integrate.quad(lambda x: x * self.fos_pdf(x, N), -np.inf, np.inf)[0]

    def exp_sos(self, N):
        return integrate.quad(lambda x: x * self.sos_pdf(x, N), -np.inf, np.inf)[0]


class DiscreteDistribution(Distribution):

    def __init__(self, dist):
        super().__init__(dist)
        self.dist = dist

    def pmf(self, x):
        return self.dist.pmf(x)

