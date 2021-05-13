import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Tabular.Agents import MonteCarloAgent
from utils import evaluate
from environments import AuctionEnvV0

REAL_MAX_EPISODE_STEPS = 100  # CUT OF AN EPISODE THAT RUNS LONGER THAN THAT. DO NOT CHANGE


def monte_carlo_eval(env, config, q_table, max_steps=REAL_MAX_EPISODE_STEPS):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env: environment to execute evaluation on
    :param config: configuration dictionary containing hyperparameters
    :param q_table: Q-table mapping observation-action to Q-values
    :param max_steps: max number of steps per evaluation episode
    :param eval_episodes: number of evaluation episodes
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = MonteCarloAgent(
        action_space=env.action_space, obs_space=env.observation_space,
        gamma=config["gamma"], epsilon=0.0)
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, max_steps, config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        returns over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table, final state-action counts
    """
    agent = MonteCarloAgent(action_space=env.action_space, obs_space=env.observation_space,
                            gamma=config["gamma"], epsilon=config["epsilon"])

    step_counter = 0
    total_eps = config["total_eps"]
    max_steps = total_eps * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in range(1, total_eps + 1):
        obs = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            act = agent.act(obs)

            n_obs, reward, done, _ = env.step(act)

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.schedule_hyperparameters(eps_num, total_eps)
        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            print(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return} ({negative_returns}/{config['eval_episodes']} failed episodes)")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


CONFIG = {
    "env": "Taxi-v3",
    "total_eps": 100000,
    "eps_max_steps": 100,
    "eval_episodes": 500,
    "eval_freq": 5000,
    "gamma": 0.99,
    "epsilon": 0.95,
}

if __name__ == "__main__":
    # env = gym.make(CONFIG["env"])
    env = AuctionEnvV0(dist=stats.geom(p=0.75), maxbudget=25, n_bidders=10, maxsteps=CONFIG["eps_max_steps"])
    total_reward, evaluation_return_means, _, q_table = train(env, CONFIG)
    print()
    print(f"Total reward over training: {total_reward}\n")
    # print("Q-table:")
    # print(q_table)
    arr = np.array(evaluation_return_means)
    plt.rc('figure', figsize=(6,2))
    plt.plot(np.arange(arr.size), arr, lw=3)
    plt.title("Monte Carlo training and evaluation")
    plt.grid(alpha=0.6)
    plt.show()
