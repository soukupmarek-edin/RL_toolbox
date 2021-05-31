import gym
import numpy as np
from approximate_methods.dqn import DQN
from utils.experience_replay import ReplayBuffer
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def play_episode(env, agent, replay_buffer,
                 train=True, explore=True, max_steps=200, batch_size=64):
    obs = env.reset()
    done = False
    losses = []
    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)

        if train:
            replay_buffer.push(np.array(obs, dtype=np.float32),
                               np.array([action], dtype=np.float32),
                               np.array(nobs, dtype=np.float32),
                               np.array([reward], dtype=np.float32),
                               np.array([done], dtype=np.float32)
                               )

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)
                losses.append(loss)

        episode_timesteps += 1
        episode_return += reward

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return, losses


def train(env, config):
    agent = DQN(env.action_space, env.observation_space,
                config["learning_rate"], config["hidden_size"],
                config["target_update_freq"], config["batch_size"],
                config["gamma"], config["epsilon"])
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    losses_all = []
    returns_all = []

    for epis in range(config["max_episodes"]):
        agent.schedule_hyperparameters(epis)
        episode_timesteps, episode_return, losses = play_episode(env, agent, replay_buffer,
                                                                 train=True, explore=True,
                                                                 max_steps=config["episode_length"],
                                                                 batch_size=config["batch_size"]
                                                                 )
        losses_all += losses
        returns_all.append(episode_return)

        if epis > 0 and (epis % config["eval_freq"]) == 0:
            eval_returns = 0
            for _ in range(config['eval_episodes']):
                _, episode_return, _ = play_episode(env, agent, replay_buffer,
                                                    train=False, explore=False,
                                                    max_steps=config["episode_length"],
                                                    batch_size=config["batch_size"]
                                                    )
                eval_returns += episode_return / config["eval_episodes"]
            eval_returns_all.append(eval_returns)
            print(f"episode: {epis}, mean return: {eval_returns:.2f}, epsilon: {agent.epsilon:.2f}")
    return np.array(eval_returns_all)


CONFIG = {
        "env": "CartPole-v1",
        "episode_length": 200,
        "max_episodes": 1500,
        "eval_freq": 5,
        "eval_episodes": 5,
        "learning_rate": 0.001,
        "hidden_size": (128, 64),
        "target_update_freq": 2500,
        "batch_size": 32,
        "gamma": 0.99,
        "epsilon": 0.7,
        "buffer_capacity": int(1e6),
    }

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    eval_returns = train(env, CONFIG)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(len(eval_returns)), eval_returns)
    plt.show()
