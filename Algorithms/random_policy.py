"""
Random policy
This file contains the method to test random policies (each time choosing a random action). The method needs the
environment and incident generator seeds.
"""

import numpy as np

NUM_STATIONS = 18


def test(env, seeds, ep_length=100):
    """
    Test a random policy.

    Parameters
    __________
    env : Environment()
        Environment where to test the policy
    seeds : list of int
        Starting points of the environment
    """

    all_rewards = []
    for init in seeds:
        with open('Results/Random_' + str(init) + '.txt', 'w') as f:
            env.reset()
            env.incidents_generator.idx = init
            _, _, _, amb_stat, _ = env.step(18)
            total_reward_random = 0
            for _ in range(ep_length):
                f.write(env.render())
                found = False
                while not found:
                    action = np.random.randint(0, 18)
                    if amb_stat[action]['num'] > 0:
                        found = True
                _, reward, _, amb_stat, _ = env.step(action)
                total_reward_random += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_random, 4)) + '\n')

        print('Reward random (' + str(init) + '): ' + str(total_reward_random))
        all_rewards.append(total_reward_random)

    print('Average reward random:', np.mean(np.array(all_rewards)))
    print('__________________________________________________')
