import numpy as np

import gym
import torch
from torch import nn, Tensor
from torch.optim import Adam

from typing import Iterable
from copy import deepcopy
from collections import namedtuple


class FCLinearNetwork(nn.Module):

    def __init__(self, dims, output_activation=None):
        super().__init__()
        self.input_size = dims[0]
        self.output_size = dims[-1]
        self.layers = self.make_seq(dims, output_activation)

    @staticmethod
    def make_seq(dims, output_activation):
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())
        mods.append(nn.ReLU())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x):
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_param(self, source, tau):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * source_param.data + tau * source_param.data)


Transition = namedtuple("Transition", ("states", "actions", "next_states", "rewards", "done"))


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = None
        self.writes = 0

    def init_memory(self, transition):
        for t in transition:
            assert t.ndim == 1

        self.memory = Transition(*[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition])

    def push(self, *args):
        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes += 1

    def sample(self, batch_size, device="cpu"):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        batch = Transition(*[torch.from_numpy(np.take(d, samples, axis=0)).to(device) for d in self.memory])

        return batch

    def __len__(self):
        return min(self.writes, self.capacity)


class DQN:

    def __init__(self, action_space, observation_space, learning_rate, hidden_size, target_update_freq, batch_size,
                 gamma, epsilon):
        self.action_space = action_space
        self.observation_space = observation_space

        ACTION_SIZE = action_space.n
        STATE_SIZE = observation_space.shape[0]

        self.online_net = FCLinearNetwork((STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None)
        self.target_net = deepcopy(self.online_net)
        self.optimizer = Adam(self.online_net.parameters(), lr=learning_rate, eps=1e-3)
        self.loss_fn = nn.MSELoss()

        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon

        self.update_counter = 0

    def act(self, obs, explore):
        obs_tensor = torch.tensor(obs).float()
        epsilon = self.epsilon if explore else 0.

        if np.random.uniform() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                Qp = self.online_net(obs_tensor)
                Q, A = torch.max(Qp, axis=0)
                return A.item()

    def update(self, batch):
        states, actions, next_states, rewards, dones = batch

        qp = self.online_net(states)
        qp_pred = self.online_net(next_states)
        q_next, _ = torch.max(qp_pred, axis=1)
        q_next = q_next.view(-1, 1)

        pred_return = torch.gather(qp, 1, torch.LongTensor(actions.tolist()))
        target_return = rewards + self.gamma * (1 - dones) * q_next

        self.optimizer.zero_grad()
        loss = self.loss_fn(target_return, pred_return)
        loss.backward()

        self.optimizer.step()

        self.update_counter += 1
        if (self.update_counter % self.target_update_freq) == 0:
            self.target_net.hard_update(self.online_net)

        return loss.item()

    def schedule_hyperparameters(self, epis):
        if self.epsilon > 0.15:
            self.epsilon -= 0.001