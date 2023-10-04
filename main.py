"""
Main to test the trained algorithms. When executing the code write the numbers of the algorithms you want to try,
e.g. '1, 2, 3, 5' to use Random, Greedy, LinearQ-Learning and Double DQN.
"""

import numpy as np

from environment import *
from Algorithms.linear_q_learning import Qlearning
from Algorithms import random_policy as random, greedy_policy as greedy, linear_q_learning as lql, dqn as dqn, smbpo as smbpo, soft_actor_critic as sac

algorithms = {1: {'name': 'Random', 'test': random.test},
              2: {'name': 'Greedy', 'test': greedy.test},
              3: {'name': 'LinearQL', 'test': lql.test},
              4: {'name': 'DQN', 'test': dqn.test},
              5: {'name': 'Double-DQN', 'test': dqn.test_double},
              6: {'name': 'SAC', 'test': sac.test},
              7: {'name': 'SMBPO', 'test': smbpo.test}}

num_trials = 5

np.random.seed(0)


if __name__ == '__main__':

    # Menu
    print('Choose algorithm to execute:')
    for alg in algorithms:
        print('\t' + str(alg) + '.- ' + algorithms[alg]['name'])
    options = input().split(', ')

    # Environment
    env = Environment()
    env.reset()
    num_incidents = env.incidents_generator.num_incidents
    np.random.seed(0)
    seeds = np.random.randint(0, num_incidents-100, num_trials)

    # Test
    for alg in options:
        np.random.seed(0)
        env.reset()
        algorithms[int(alg)]['test'](env, seeds)




