import gym
import numpy as np
from approximate_methods.policy_gradient import Reinforce
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def play_episode(env, agent, train=True, explore=True, max_steps=200):
    obs = env.reset()

    done = False
    num_steps = 0
    episode_return = 0

    observations = []
    actions = []
    rewards = []
    losses = []

    while not done and num_steps < max_steps:
        action = agent.act(np.array(obs), explore=explore)
        nobs, reward, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        num_steps += 1
        episode_return += reward

        obs = nobs

    if train:
        loss = agent.update(rewards, observations, actions)
        losses.append(loss)

    return num_steps, episode_return, losses


def train(env, config):
    agent = Reinforce(env.action_space, env.observation_space,
                      config["learning_rate"], config["hidden_size"],
                      config["gamma"])

    eval_returns_all = []
    losses_all = []
    returns_all = []

    for epis in range(config["max_episodes"]):
        agent.schedule_hyperparameters(epis)
        episode_steps, episode_return, losses = play_episode(env, agent,
                                                             train=True, explore=True,
                                                             max_steps=config["episode_length"])
        losses_all += losses
        returns_all.append(episode_return)

        if epis > 0 and (epis % config["eval_freq"]) == 0:
            eval_returns = 0
            for _ in range(config['eval_episodes']):
                _, episode_return, _ = play_episode(env, agent,
                                                    train=False, explore=False,
                                                    max_steps=config["episode_length"]
                                                    )
                eval_returns += episode_return / config["eval_episodes"]
            eval_returns_all.append(eval_returns)
            print(f"episode: {epis}, mean return: {eval_returns:.2f}")
    return np.array(eval_returns_all)


CONFIG = {
        "env": "CartPole-v1",
        "episode_length": 200,
        "max_episodes": 2000,
        "eval_freq": 20,
        "eval_episodes": 5,
        "learning_rate": 0.01,
        "hidden_size": (16, 16),
        "gamma": 0.99,
    }

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])

    eval_returns = train(env, CONFIG)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(len(eval_returns)), eval_returns)
    plt.show()
