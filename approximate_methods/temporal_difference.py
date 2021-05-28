from abc import ABC
import numpy as np
import gym

from sklearn.linear_model import SGDRegressor


class LinearEstimator:

    def __init__(self, act_dim, init_obs, eta, scaler, featurizer):
        self.scaler = scaler
        self.featurizer = featurizer

        self.models = []
        for _ in range(act_dim):
            model = SGDRegressor(learning_rate="constant", eta0=eta)
            model.partial_fit([self.featurize_state(init_obs)], [0])
            self.models.append(model)

    def featurize_state(self, obs):
        scaled = self.scaler.transform([obs])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, obs):
        features = self.featurize_state(obs)
        return np.array([m.predict([features])[0] for m in self.models])

    def update(self, obs, act, y):
        features = self.featurize_state(obs)
        self.models[act].partial_fit([features], [y])


class QLearningAgent:

    def __init__(self,
                 action_space: gym.Space, observation_space: gym.Space,
                 estimator: LinearEstimator,
                 gamma: float, epsilon: float):

        self.action_space = action_space
        self.act_dim = action_space.n
        self.obs_space = observation_space
        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = estimator

    def act(self, obs: np.ndarray, explore=False) -> float:
        eps = self.epsilon if explore else 0
        act_probs = np.ones(self.act_dim) * eps / self.act_dim

        a = self.policy.predict(obs).argmax()
        act_probs[a] += (1.-eps)
        return np.random.choice(np.arange(self.act_dim), p=act_probs)

    def learn(self, obs, action, n_obs, reward, done):

        qval_next = self.policy.predict(n_obs).max()
        td_target = reward + self.gamma * qval_next
        self.policy.update(obs, action, td_target)

    def schedule_hyperparameters(self, **kwargs):
        eps_min = kwargs['eps_min']
        eps_decay = kwargs['eps_decay']

        if self.epsilon > eps_min:
            self.epsilon -= eps_decay



