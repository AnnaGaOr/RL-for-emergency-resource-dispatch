"""
Greedy policy: send the nearest available ambulance
"""

import numpy as np

NUM_STATIONS = 18


class GreedyPolicy:
    """
    Class to apply the greedy policy

    Methods
    _______
    choose_action(obs, amb_stat)
        choose action according to the greedy policy
    """

    def __init__(self):
        pass
    
    def choose_action(self, obs, amb_stat):
        """
        Choose action according to a greedy policy: send the nearest available ambulance

        Parameters
        __________
        obs : tuple (time : int, priority : int, coordinates : tuple (float, float), distances : list (len NUM_STATIONS
                     of float))
            observation of the next state returned by the environment after a step
        amb_stat : dict
            state of the ambulances at the time of the ambulances

        Returns
        _______
        int
            action
        """
        distances = obs[3]
        order = np.argsort(distances)
        idx = 0

        action_found = False
        while not action_found:
            if amb_stat[order[idx]]['num'] > 0:
                action = order[idx]
                action_found = True
            idx += 1

        return action


def test(env, seeds, ep_length=100):
    """
    Test the policy

    Parameters
    __________
    env : environment.CustomEnv()
        environment
    seeds : list (of int)
        starting indices incident generator to calculate rewards

    Return
    ______
    float
        average reward with all seeds
    """

    greedy = GreedyPolicy()
    all_rewards = []
    for init in seeds:
        with open('Results/Greedy_' + str(init) + '.txt', 'w') as f:
            env.reset()
            env.incidents_generator.idx = init
            obs, _, _, amb_stat, _ = env.step(18)
            total_reward_greedy = 0
            for i in range(ep_length):
                f.write(env.render())
                action = greedy.choose_action(obs, amb_stat)
                obs, reward, _, amb_stat, _ = env.step(action)
                total_reward_greedy += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_greedy, 4)) + '\n')

        print('Reward greedy (' + str(init) + '): ' + str(total_reward_greedy))
        all_rewards.append(total_reward_greedy)

    print('Average reward greedy:', np.mean(np.array(all_rewards)))
    print('__________________________________________________')
