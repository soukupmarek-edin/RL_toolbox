from scipy import stats
import numpy as np

from utils import ContinuousDistribution, DiscreteDistribution
from MDP.mdp_solver import ValueIteration

# # searching robot environment
# state_names = ['high', 'low']
# action_names = ['wait', 'recharge', 'search']
# P = np.array([[[1, 0], [1, 0], [0.8, 0.2]], [[0, 1], [1, 0], [0.6, 0.4]]])
# R = np.array([[[2, 0], [0, 0], [5, 5]], [[0, 2], [0, 0], [-3, 5]]])



# bidding MDP
n_bidders = 10
dst = stats.lognorm(s=0.5, loc=0, scale=np.exp(1))
dst = stats.uniform(0,10)
dist = ContinuousDistribution(dst)

dst = stats.geom(p=0.3)
dist = DiscreteDistribution(dst)

print(dist.exp_fos(n_bidders))

state_names = ['win', 'lose']
budget = 10
states = np.arange(budget+1)
actions = np.arange(budget+1)

P = np.zeros((states.size, actions.size, states.size))
R = np.zeros((states.size, actions.size, states.size))

for s in range(states.size):
    for a in range(actions.size):
        if s >= a:
            P[s, a, s-a] = dist.fos_cdf(a, n_bidders)
            P[s, a, s] = 1-dist.fos_cdf(a, n_bidders)
        else:
            P[s, a, s] = 1

        if s >= a:
            R[s, a, s-a] = 1
            R[s, 0, s] = 0

if __name__ == "__main__":
    gamma = 0.99
    theta = 1e-6

    vi = ValueIteration(P, R, gamma=gamma)
    policy, V = vi.solve(theta=theta)
    print("policy:\n", policy)
    print("\nvalue function:\n", V)