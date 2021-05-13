from abc import ABC, abstractmethod
import gym
import numpy as np
from typing import Dict, Iterable, List
from scipy.special import softmax
from scipy import stats
import torch
import torch.nn
from torch.optim import Adam
from networks import FCNetwork
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Agent(ABC):
    """
    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class Reinforce(Agent):
    """
    :attr policy (FCNetwork): fully connected actor network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for actor network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #

        ### DO NOT CHANGE THE OUTPUT ACTIVATION OF THIS POLICY ###
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = kwargs['epsilon_start']
        self.epsilon_end = kwargs['epsilon_end']
        self.epsilon_decline = kwargs['epsilon_decline']
        self.epsilon = kwargs['epsilon']

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        start_eps = self.epsilon_start
        end_eps = self.epsilon_end
        s = self.epsilon_decline
        if self.epsilon > 0.02:
            self.epsilon = start_eps - (start_eps-end_eps) / (1 + np.exp(-s * (timestep - max_timesteps / 2)))

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        obs_tensor = torch.tensor(obs).float()
        epsilon = self.epsilon if explore else 0.

        if np.random.uniform() < epsilon:
            return self.action_space.sample()
        else:
            dist = self.policy(obs_tensor).detach().numpy()
            return np.random.choice(np.arange(self.action_space.n), p=dist)

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        r = np.array([self.gamma ** i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        r = r - r.mean()

        observations_tensor = torch.FloatTensor(observations)
        rewards_tensor = torch.FloatTensor(r)
        actions_tensor = torch.LongTensor(actions)

        pred = self.policy(observations_tensor) + 1e-5
        logprob = torch.log(pred)
        # print(pred)
        selected_logprobs = rewards_tensor * torch.gather(logprob, 1, actions_tensor.unsqueeze(1)).squeeze()

        p_loss = -selected_logprobs.mean()
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        return {"p_loss": p_loss}


class GaussianReinforce(Agent):
    """
    :attr policy (FCNetwork): fully connected actor network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for actor network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(self, action_space: gym.Space, obs_space: gym.Space, d_features,
                 alpha: float, gamma: float, epsilon: float, sigma=1):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, obs_space)
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

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

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
