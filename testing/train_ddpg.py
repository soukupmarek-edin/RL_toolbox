import gym
import numpy as np
import matplotlib.pyplot as plt
from approximate_methods.policy_gradient import DDPG
from utils.experience_replay import ReplayBuffer


def play_episode(env, agent, replay_buffer, train=True, explore=True, max_steps=200, batch_size=64):
    obs = env.reset()
    done = False

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)

        episode_timesteps += 1
        episode_return += reward

        if max_steps == episode_timesteps:
            break
        obs = nobs
    return episode_timesteps, episode_return


def train(env, config):
    agent = DDPG(action_space=env.action_space, observation_space=env.observation_space, **config)
    replay_buffer = ReplayBuffer(config["buffer_capacity"])
    eval_returns_all = []

    for epis in range(config['max_episodes']):
        agent.schedule_hyperparameters(epis, config["max_episodes"])
        episode_timesteps, _ = play_episode(env, agent, replay_buffer,
                                            train=True, explore=True,
                                            max_steps=config["episode_length"], batch_size=config["batch_size"])

        if (epis % config["eval_freq"]) == 0 and epis > 0:
            eval_returns = 0
            for _ in range(config["eval_episodes"]):
                _, episode_return = play_episode(env, agent, replay_buffer,
                                                 train=False, explore=False,
                                                 max_steps=config["episode_length"], batch_size=config["batch_size"])

                eval_returns += episode_return / config["eval_episodes"]
            eval_returns_all.append(eval_returns)
            print(f"episode: {epis}, mean return: {eval_returns:.2f}, tau: {agent.tau:.3f}")

    return np.array(eval_returns_all)


### TUNE HYPERPARAMETERS HERE ###
PENDULUM_CONFIG = {
    "env": "Pendulum-v0",
    # "env": "MountainCarContinuous-v0",
    "target_return": -10.0,
    "episode_length": 200,
    "max_episodes": 500,
    "eval_freq": 10,
    "eval_episodes": 5,
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "actor_hidden_size": [64, 64],
    "critic_hidden_size": [64, 64],
    "tau": 0.001,
    "tau_start": 0.001,
    "tau_end": 0.001,
    "tau_change": 0.05,
    "batch_size": 16,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "save_filename": "pendulum_latest.pt",
}


CONFIG = PENDULUM_CONFIG


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    eval_returns_all = train(env, CONFIG)
    env.close()

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(len(eval_returns_all)), eval_returns_all)
    plt.show()