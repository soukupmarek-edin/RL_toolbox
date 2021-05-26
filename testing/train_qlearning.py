import gym
from tabular.Agents import QLearningAgent, SarsaAgent
from utils import evaluate, DiscreteDistribution, ContinuousDistribution
import environments as my_envs
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


MAX_EVAL_EPS_STEPS = 100


def q_learning_eval(env, config, q_table, max_steps):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env: environment to execute evaluation on
    :param config: configuration dictionary containing hyperparameters
    :param q_table: Q-table mapping observation-action to Q-values
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(action_space=env.action_space, obs_space=env.observation_space,
                                gamma=config["gamma"], alpha=config["alpha"], epsilon=0.0)
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, max_steps, config["eval_episodes"])


def train(env, config):
    agent = QLearningAgent(action_space=env.action_space, obs_space=env.observation_space,
                           gamma=config["gamma"], alpha=config["alpha"], epsilon=config["epsilon"])
    total_eps = config["total_eps"]
    eps_max_steps = config["eps_max_steps"]
    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []
    obs_counter = np.zeros(env.observation_space.n)
    episode_lens = np.zeros(config["eps_max_steps"]+1)

    for eps_num in range(total_eps):
        obs = env.reset()
        episodic_return = 0
        t = 0

        while t < eps_max_steps:
            obs_counter[obs] += 1
            action = agent.act(obs)
            n_obs, reward, done, _ = env.step(action)
            agent.learn(obs, action, n_obs, reward, done)

            episodic_return += reward
            t += 1
            if done:
                break
            obs = n_obs

        agent.schedule_hyperparameters(eps_num, total_eps)
        total_reward += episodic_return
        episode_lens[t] += 1

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            env.randomize_maxbudget = False
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table, MAX_EVAL_EPS_STEPS)
            print(f"alpha = {agent.alpha:.2f}, epsilon = {agent.epsilon:.2f}")
            print(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return} ({negative_returns}/{config['eval_episodes']} failed episodes)")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)
            env.randomize_maxbudget = True

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table, obs_counter, episode_lens


### TUNE HYPERPARAMETERS HERE ###
CONFIG = {
    "env": "Taxi-v3",
    "total_eps": 100000,
    "eps_max_steps": 50,
    "eval_episodes": 500,
    "eval_freq": 1000,
    "gamma": 0.99,
    "alpha": 0.15,
    "epsilon": 0.9,
}

if __name__ == "__main__":
    # env = gym.make(CONFIG["env"])
    dist = ContinuousDistribution(stats.lognorm(s=0.5, loc=0, scale=np.exp(0.25)))
    n_bidders = 10
    env = my_envs.AuctionEnvV1(dist, maxbudget=25, n_bidders=n_bidders, maxsteps=CONFIG["eps_max_steps"], action_granularity=0.25)
    total_reward, evaluation_return_means, evaluation_negative_returns, q_table, obs_counter, episode_lens = train(env, CONFIG)
    np.save("Qlearning_qtable.npy", q_table)
    np.save("Qlearning_obscount.npy", obs_counter)
    np.save("Qlearning_epslens.npy", episode_lens)
    print()
    print(f"Total reward over training: {total_reward}\n")
    arr = np.array(evaluation_return_means)

    fig = plt.figure(figsize=(6,5))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313, sharex=ax2)

    ax1.plot(np.arange(arr.size), arr, lw=3, label="mean evaluation reward")
    ax1.axhline(env.maxbudget/dist.exp_fos(n_bidders), color='k', ls='dashed', label='expected highest bid')
    ax1.set_title("Q-learning training and evaluation")
    ax1.set_xlabel("evaluation step")
    ax1.legend()

    ax2.plot(np.linspace(0, env.maxbudget, env.observation_space.n), q_table.argmax(axis=1) * env.s, lw=2.5, label='estimated bidding function')
    ax2.axhline(dist.exp_fos(n_bidders), color='k', ls='dashed', label='expected highest bid')
    ax2.set_xlabel("budget")
    ax2.set_ylabel("optimal bid")
    ax2.legend()

    ax3.bar(np.linspace(0, env.maxbudget, env.observation_space.n), obs_counter, edgecolor='k', width=0.25)
    ax3.set_xlabel('budget')
    ax3.set_ylabel('number of visits')
    for ax in [ax1, ax2, ax3]:
        ax.grid(alpha=0.4)
    plt.savefig("Qlearning_results_latest.jpg")
    plt.show()
