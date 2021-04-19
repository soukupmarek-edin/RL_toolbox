import numpy as np
import matplotlib.pyplot as plt


class Bandit:

    def __init__(self, n_actions, n_steps, reward_estimates_init, step_size):
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.step_size = step_size

        self.n_action_selected = np.ones(n_actions) / 1e5
        self.reward_estimates = reward_estimates_init
        self.preferences = np.zeros(n_actions)
        self.action = 0
        self.step = 1

    def update_reward_estimate(self, new_reward):
        current_reward = self.reward_estimates[self.action]

        if self.step_size == 'mean':
            n_action = self.n_action_selected[self.action]
            alpha = 1 / n_action
        else:
            alpha = self.step_size

        self.reward_estimates[self.action] = current_reward + alpha * (new_reward - current_reward)
        self.step += 1


class EpsilonGreedyBandit(Bandit):

    def __init__(self, n_actions, n_steps, reward_estimates_init, step_size, epsilon=0):
        super().__init__(n_actions, n_steps, reward_estimates_init, step_size)
        self.epsilon = epsilon

    def select_action(self):
        rnd = np.random.uniform()
        if self.epsilon > rnd:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = self.reward_estimates.argmax()

        self.n_action_selected[action] += 1
        self.action = action
        return action


class UCBBandit(Bandit):

    def __init__(self, n_actions, n_steps, reward_estimates_init, step_size, c=2):
        super().__init__(n_actions, n_steps, reward_estimates_init, step_size)
        self.c = c

    def select_action(self):
        action = np.argmax(self.reward_estimates + self.c * np.sqrt(np.log(self.step) / self.n_action_selected))

        self.n_action_selected[action] += 1
        self.action = action
        return action


class GradientBandit(Bandit):
    def __init__(self, n_actions, n_steps, reward_estimates_init, step_size, use_baseline):
        super().__init__(n_actions, n_steps, reward_estimates_init, step_size)
        self.H = np.zeros(n_actions)
        self.policy = self.softmax()
        self.use_baseline = use_baseline
        self.reward_avg = 0
        self.cnt = 0

    def select_action(self):
        self.policy = self.softmax()
        action = np.random.choice(np.arange(self.n_actions), p=self.policy)
        self.n_action_selected[action] += 1
        self.action = action
        return action

    def softmax(self):
        h = self.H
        return np.exp(h)/np.sum(np.exp(h))

    def update_preferences(self, reward):
        alpha = 1 / self.n_action_selected[self.action] if self.step_size == 'mean' else self.step_size
        a = self.action

        self.cnt += 1
        if self.use_baseline:
            self.reward_avg = self.reward_avg + 1/self.cnt*(reward-self.reward_avg)
            reward_avg = self.reward_avg
        else:
            reward_avg = 0

        self.H[a] = self.H[a] + alpha * (reward - reward_avg) * (1-self.policy[a])
        self.H[:a] = self.H[:a] - alpha * (reward - reward_avg) * self.policy[:a]
        self.H[a+1:] = self.H[a+1:] - alpha * (reward - reward_avg) * self.policy[a+1:]


class Task:

    def __init__(self, bandit_type, n_actions, n_steps, rewards, reward_estimates_init, step_size, **kwargs):
        self.bandit_type = bandit_type
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.rewards = rewards

        self.optimal_action = self.rewards.argmax()
        self.step = 1

        if bandit_type == 'epsilon_greedy':
            self.bandit = EpsilonGreedyBandit(n_actions, n_steps, reward_estimates_init, step_size, **kwargs)
        elif bandit_type == 'upper_confidence_bound':
            self.bandit = UCBBandit(n_actions, n_steps, reward_estimates_init, step_size, **kwargs)
        elif bandit_type == 'gradient':
            self.bandit = GradientBandit(n_actions, n_steps, reward_estimates_init, step_size, **kwargs)
        else:
            raise AttributeError("Unknown bandit type")

        self.reward_tracker = np.zeros(n_steps)
        self.optimal_action_tracker = np.zeros(n_steps)

    def make_step(self):
        action = self.bandit.select_action()
        reward = self.rewards[action] + np.random.normal()
        self.bandit.update_reward_estimate(reward)
        if self.bandit_type == 'gradient':
            self.bandit.update_preferences(reward)
        elif self.bandit_type in ['epsilon_greedy', 'upper_confidence_bound']:
            self.bandit.update_reward_estimate(reward)

        # trackers
        self.reward_tracker[self.step - 1] = reward
        if action == self.optimal_action:
            self.optimal_action_tracker[self.step - 1] += 1

    def run_task(self):
        for _ in range(self.n_steps):
            self.make_step()
            self.step += 1


if __name__ == '__main__':
    n_actions = 10
    n_steps = 1000
    n_runs = 500

    task_rewards = np.zeros((n_steps, n_runs))
    optimal_actions = np.zeros((n_steps, n_runs))

    plt.rc('figure', figsize=(12, 5))
    fig, ax = plt.subplots()

    # for step_size in [0.1, 0.4]:
    #     for i in range(n_runs):
    #         task = Task('gradient', n_actions, n_steps,
    #                     rewards=np.random.normal(4, size=n_actions),
    #                     reward_estimates_init=np.zeros(n_actions),
    #                     step_size=step_size,
    #                     use_baseline=True)
    #         task.run_task()
    #         optimal_actions[:, i] = task.optimal_action_tracker
    #
    #     ax.plot(optimal_actions.mean(axis=1), label=f"alpha = {step_size}")

    for step_size in [0.1, 0.4]:
        for i in range(n_runs):
            task = Task('gradient', n_actions, n_steps,
                        rewards=np.random.normal(4, size=n_actions),
                        reward_estimates_init=np.zeros(n_actions),
                        step_size=step_size,
                        use_baseline=False)
            task.run_task()
            optimal_actions[:, i] = task.optimal_action_tracker

        ax.plot(optimal_actions.mean(axis=1), label=f"alpha = {step_size}, baseline = False")

    ax.grid()
    ax.legend()
    ax.set_ylim((0, 1))
    plt.show()