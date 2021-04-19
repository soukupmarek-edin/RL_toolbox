import numpy as np


class MonteCarloAgent:
    """Agent using the Monte-Carlo algorithm for training
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = defaultdict(lambda: 0)
        self.returns = defaultdict(lambda: 0)

        self.Q_table = np.array()

    def learn(self, observations, actions, rewards):
        updated_values = {}

        G = 0
        T = len(obses)
        sa_pairs = list(zip(obses, actions))
        for t in list(range(T))[::-1]:
            G = self.gamma*G + rewards[t]
            St, At = obses[t], actions[t]
            if (St, At) not in sa_pairs[:t]:
                self.returns[(St, At)] += G
                self.sa_counts[(St, At)] += 1
                self.q_table[(St, At)] = self.returns[(St, At)] / self.sa_counts[(St, At)]
                updated_values[(St, At)] = self.q_table[(St, At)]

        return updated_values

    def schedule_hyperparameters(self, timestep, max_timestep):
        pass