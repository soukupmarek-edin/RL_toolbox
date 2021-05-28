import numpy as np
import matplotlib.pyplot as plt

import gym
from approximate_methods.temporal_difference import LinearEstimator, QLearningAgent

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


MAX_EVAL_EPS_STEPS = 100


def play_episode(env, agent, config, explore=True, learn=True):
    episodic_return = 0
    obs = env.reset()
    t = 0

    while t < config['epis_max_steps']:
        action = agent.act(obs, explore=explore)
        n_obs, reward, done, _ = env.step(action)

        if learn:
            agent.learn(obs, action, n_obs, reward, done)

        episodic_return += reward
        t += 1
        if done:
            break
        obs = n_obs

    return agent, episodic_return


def train(env, config):
    agent = QLearningAgent(env.action_space, env.observation_space, estimator, config['gamma'], config['epsilon'])
    total_epis = config['total_epis']

    evaluation_return_means = []
    total_reward = 0

    for epis_num in range(total_epis):
        agent, epis_return = play_episode(env, agent, config, explore=True, learn=True)
        agent.schedule_hyperparameters(**config)
        total_reward += epis_return

        if epis_num > 0 and epis_num % config['eval_freq'] == 0:
            eval_returns = []
            for t in range(config['eval_episodes']):
                _, epis_return = play_episode(env, agent, config, explore=False, learn=False)
                eval_returns.append(epis_return)

            evaluation_return_means.append(np.mean(eval_returns))
            print(f"EVALUATION: EP {epis_num} - MEAN RETURN: {evaluation_return_means[-1]}, epsilon: {agent.epsilon:.2f}")

    return total_reward, evaluation_return_means


### TUNE HYPERPARAMETERS HERE ###
CONFIG = {
    "env": "MountainCar-v0",
    "total_epis": 2000,
    "epis_max_steps": 1000,
    "eval_episodes": 100,
    "eval_freq": 100,
    "gamma": 0.99,
    "eta": 0.15,
    "epsilon": 0.95,
    "eps_min": 0.2,
    "eps_decay": (3/2)/2000*(0.95-0.2)
}

if __name__ == "__main__":
    env = gym.envs.make("MountainCar-v0")

    init_obs = 10000
    observations = np.array([env.observation_space.sample() for _ in range(init_obs)])

    scaler = StandardScaler()
    scaler.fit(observations)

    featurizer = pipeline.FeatureUnion([('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                                        ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                                        ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                                        ('rbf4', RBFSampler(gamma=0.5, n_components=100))
                                        ])
    featurizer.fit(scaler.transform(observations))
    estimator = LinearEstimator(env.action_space.n, env.reset(), CONFIG['eta'], scaler, featurizer)

    total_reward, eval_mean_returns = train(env, CONFIG)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(eval_mean_returns)
    plt.show()

