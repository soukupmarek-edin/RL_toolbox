import gym
import numpy as np
from typing import Dict, Iterable, List
from scipy import stats
import torch
from torch.optim import Adam
import os
from torch.distributions import Normal
from torch.autograd import Variable
from utils.networks import FCNetwork

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DDPG:
    """
    off-policy actor-critic
    """

    def __init__(self, action_space, observation_space, gamma,
                 actor_learning_rate, critic_learning_rate, actor_hidden_size, critic_hidden_size, tau, **kwargs):
        self.action_space = action_space
        self.observation_space = observation_space
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.actor = FCNetwork((STATE_SIZE, *actor_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh)
        self.actor_target = FCNetwork((STATE_SIZE, *actor_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh)
        self.actor_target.hard_update(self.actor)

        self.critic = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None)
        self.critic_target = FCNetwork((STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None)
        self.critic_target.hard_update(self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=actor_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)

        self.critic_criterion = torch.nn.MSELoss()

        self.gamma = gamma
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        self.tau_start = kwargs['tau_start']
        self.tau_end = kwargs['tau_end']
        self.tau_change = kwargs['tau_change']

        self.noise_dist = Normal(0, 0.1)

    def schedule_hyperparameters(self, epis, max_episodes):
        if self.tau >= self.tau_end:
            self.tau = self.tau - self.tau_change * (self.tau_start - self.tau_end)

    def act(self, obs, explore=True):
        obs_tensor = Variable(torch.from_numpy(obs).float().unsqueeze(0))
        if explore:
            action = self.actor(obs_tensor) + self.noise_dist.sample()
        else:
            action = self.actor(obs_tensor)

        return np.clip(action.detach().numpy()[0], -1, 1)

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch

        # Critic loss
        features = torch.cat([actions, states], dim=1)

        Qvals = self.critic.forward(features)
        next_actions = self.actor_target.forward(next_states)
        features = torch.cat([next_actions, next_states], dim=1)

        next_Q = self.critic_target.forward(features)
        Qprime = rewards + self.gamma*(1-dones)*next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        states_pred = self.actor.forward(states)
        features = torch.cat([states_pred, states], dim=1)
        actor_loss = -self.critic.forward(features).mean()

        # update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.actor_target.soft_update(self.actor, tau=self.tau)
        self.critic_target.soft_update(self.critic, tau=self.tau)

        q_loss = critic_loss
        p_loss = actor_loss
        return {"q_loss": q_loss,
                "p_loss": p_loss}


class VanillaAC:

    def __init__(self,
                 action_space: gym.Space, observation_space: gym.Space,
                 learning_rate: float, hidden_size: Iterable[int],
                 gamma: float, **kwargs):

        self.action_space = action_space
        self.observation_space = observation_space
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        self.policy = FCNetwork((STATE_SIZE, *hidden_size, ACTION_SIZE),
                                output_activation=torch.nn.modules.activation.Softmax)

        self.critic = FCNetwork((STATE_SIZE, *hidden_size, ACTION_SIZE),
                                output_activation=torch.nn.modules.activation.Softmax)

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)


        self.learning_rate = learning_rate
        self.gamma = gamma

    def schedule_hyperparameters(self, epis, max_epis):
        pass

    def act(self, obs: np.ndarray, explore: bool):

        obs_tensor = Variable(torch.tensor(obs).float())
        dist = self.policy(obs_tensor).detach().numpy()
        return np.random.choice(np.arange(self.action_space.n), p=dist)

    def discounted_rewards(self, last_value, rewards, dones):
        rtg = np.zeros_like(rewards, dtype=np.float32)
        rtg[-1] = rewards[-1] + self.gamma * last_value
        for i in reversed(range(len(rewards) - 1)):
            rtg[i] = self.rews[i] + self.gamma * rtg[i + 1]
        return rtg

    def update(self, rewards, observations, actions, last_value):

        next_value = self.discounted_rewards(rewards)

        observations_tensor = torch.FloatTensor(observations)
        rewards_tensor = torch.FloatTensor(r)
        actions_tensor = torch.LongTensor(actions)

        pred = self.policy(observations_tensor)
        logprob = torch.log(pred)
        selected_logprobs = rewards_tensor * torch.gather(logprob, 1, actions_tensor.unsqueeze(1)).squeeze()

        p_loss = -selected_logprobs.mean()
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return p_loss


class Reinforce:
    """
    :attr policy (FCNetwork): fully connected actor network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for actor network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(self,
                 action_space: gym.Space, observation_space: gym.Space,
                 learning_rate: float, hidden_size: Iterable[int],
                 gamma: float, **kwargs):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        self.action_space = action_space
        self.observation_space = observation_space
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        self.policy = FCNetwork((STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax)

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        self.learning_rate = learning_rate
        self.gamma = gamma

    def schedule_hyperparameters(self, epis, max_epis):
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """

        obs_tensor = Variable(torch.tensor(obs).float())
        dist = self.policy(obs_tensor).detach().numpy()
        return np.random.choice(np.arange(self.action_space.n), p=dist)

    def discounted_rewards(self, rewards, demean=True):
        r = np.array([self.gamma ** i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        if demean:
            return r - r.mean()
        else:
            return r

    def update(self, rewards: List[float], observations: List[np.ndarray], actions: List[int]) -> Dict[str, float]:
        """Update function for policy gradients

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values losses
        """
        r = self.discounted_rewards(rewards)

        observations_tensor = torch.FloatTensor(observations)
        rewards_tensor = torch.FloatTensor(r)
        actions_tensor = torch.LongTensor(actions)

        pred = self.policy(observations_tensor)
        logprob = torch.log(pred)
        selected_logprobs = rewards_tensor * torch.gather(logprob, 1, actions_tensor.unsqueeze(1)).squeeze()

        p_loss = -selected_logprobs.mean()
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return p_loss


class GaussianReinforce:

    def __init__(self, action_space: gym.Space, observation_space: gym.Space, d_features,
                 alpha: float, gamma: float, epsilon: float, sigma=1):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.d_features = d_features
        self.weights = np.ones(d_features+1)*0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.sigma = sigma

        self.policy = lambda mu: stats.norm(loc=mu, scale=sigma)

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        start_eps = self.epsilon_start
        end_eps = self.epsilon_end
        s = self.epsilon_decline
        if self.epsilon > 0.02:
            self.epsilon = start_eps - (start_eps-end_eps) / (1 + np.exp(-s * (timestep - max_timesteps / 2)))

    def act(self, obs: np.ndarray, explore: bool = False):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs: observation vector from the environment
        :param explore: flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        epsilon = self.epsilon if explore else 0.
        # features and bias
        feature_vector = np.concatenate([obs, np.array([1])])
        mu = np.dot(self.weights, feature_vector)

        if np.random.uniform() < epsilon:
            action = self.policy(mu).rvs()
        else:
            action = self.policy(mu).expect()
        return action

    def update(self, rewards: List[float], obses: List[np.ndarray], actions: List[int]) -> Dict[str, float]:
        """Update function for policy gradients

        :param rewards: rewards of episode (from first to last)
        :param obses: observations of episode (from first to last)
        :param actions: applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        rewards = np.array(rewards)
        obses = np.array(obses)
        actions = np.array(actions)
        T = len(obses)
        for t in range(T):
            G = np.sum([self.gamma**(k-t-1)*rewards[k] for k in range(t, T)])
            feature_vector = np.concatenate([obses[t], np.array([1])])
            mu = np.dot(self.weights, feature_vector)
            grad = (actions[t]-mu)*feature_vector/self.sigma**2
            self.weights = self.weights + self.alpha*self.gamma**t*G*grad
