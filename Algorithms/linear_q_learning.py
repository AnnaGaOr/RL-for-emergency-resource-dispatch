"""
Q-learning with linear function approximation
Algorithm of Q-Learning approximating Q with a linear function (matrix).
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

from environment import *

NUM_STATIONS = 18
SHAPE_STATE = 6 * NUM_STATIONS  # (amb per station, distance new station) * 3 priorities


class Qlearning:
    """
    Q-learning with linear function approximation

    Attributes
    __________
    epsilon : float (0 < epsilon < 1)
        probability of a random action in epsilon-greedy policy (default 0.1)
    alpha : float (0 < gamma < 1)
        learning rate for the update of the parameters (default = 0.01)
    gamma : float (0 < gamma < 1)
        discount factor for the long term reward (default = 0.9)
    env : environment.CustomEnv()
        environment
    initial_amb : list (of int, len NUM_STATIONS)
        initial number of ambulances per station
    feature_vector : numpy.array() (shape=(NUM_STATIONS+1, SHAPE_STATE))
    losses :  list (of float)
        list of losses to plot the evolution

    Methods
    _______
    action_epsilon_greedy(state)
        Samples action following the epsilon greedy policy
    random_action(state)
        Samples a random action
    policy_action(state)
        Action according to the learned policy
    state_encoding(obs, amb_stat_dict)
        Encodes the state to be entered in the NN
    learn(steps=1000)
        Linear Q-Learning algorithm to learn a policy
    plot_loss(save=None)
        Save plot loss
    """

    def __init__(self,  env):
        """
        Parameters
        __________
        env : environment.CustomEnv()
            environment
        """

        # Parameters
        self.epsilon = 0.1
        self.alpha = 0.01
        self.gamma = 0.9

        # Environment and initial ambulances in stations
        self.env = env
        amb_dict = self.env.ambu_in_stations
        self.initial_amb = []
        for j in range(NUM_STATIONS):
            self.initial_amb.append(amb_dict[j]['num'])

        # Approximating function
        self.feature_vector = np.zeros((NUM_STATIONS + 1, SHAPE_STATE))  # Q(s,a) = a^T * fv * s

        self.losses = []

    def action_epsilon_greedy(self, state):
        """
        Samples action following the epsilon greedy policy (prob epsilon random action, prob 1-epsilon policy action)

        Parameters
        __________
        state : numpy.array() (shape=(SHAPE_STATE,))
            Encoding of the observation

        Returns
        _______
        int
            action
        """

        if np.random.uniform(0, 1) < self.epsilon:
            action, value = self.random_action(state)
        else:
            action, value = self.policy_action(state)

        return action, value

    def random_action(self, state):
        """
        Samples a random action

        Parameters
        __________
        state : numpy.array() (shape=(SHAPE_STATE,))
            Encoding of the observation

        Returns
        -------
        int
            random action
        """

        # Sample a random action until it finds a possible one
        found = False
        while not found:
            act = np.random.randint(0, NUM_STATIONS + 1)
            if act == NUM_STATIONS or state[act]+state[2*NUM_STATIONS+act]+state[4*NUM_STATIONS+act] > 0:
                found = True

        # Estimated Q-value of the action
        value = np.matmul(self.feature_vector[act], state)

        return act, value

    def policy_action(self, state):
        """
        Decides the best possible action in a state according to the learned policy

        Parameters
        __________
        state : numpy.array() (shape=(SHAPE_STATE,))
            Encoding of the observation

        Returns
        int
            action
        """

        rewards_actions = np.matmul(self.feature_vector, state)
        order = np.argsort(rewards_actions)

        found = False
        i = NUM_STATIONS
        while not found:
            act = order[i]
            if act == NUM_STATIONS or state[act]+state[2*NUM_STATIONS+act]+state[4*NUM_STATIONS+act] > 0:
                found = True
            i -= 1

        return act, rewards_actions[act]

    def state_encoding(self, obs, amb_stat_dict):
        """
        Encodes the state to be multiplied with the feature vector

        Parameters
        __________
        obs : tuple (time : int, priority : int, coordinates : tuple (float, float), distances : list (len NUM_STATIONS
                     of float))
            observation of the next state returned by the environment after a step
        amb_stat_dict : dict
            state of the ambulances at the time of the ambulances

        Returns
        _______
        numpy.array() (shape=(SHAPE_STATE,))
            Encoding of the observation
        """

        priority = int(obs[1])
        distances = np.array(obs[3])
        distances = np.exp(-(distances / 1000) ** 2)

        amb_stat = []
        for j in range(NUM_STATIONS):
            amb_stat.append(amb_stat_dict[j]['num'] / self.initial_amb[j])

        state = np.zeros(6 * NUM_STATIONS)
        if priority == 0:
            state[:NUM_STATIONS] = amb_stat
            state[NUM_STATIONS:2*NUM_STATIONS] = distances
        elif priority == 1:
            state[2*NUM_STATIONS:3*NUM_STATIONS] = amb_stat
            state[3*NUM_STATIONS:4 * NUM_STATIONS] = distances
        else:
            state[4*NUM_STATIONS:5*NUM_STATIONS] = amb_stat
            state[5*NUM_STATIONS:] = distances

        return state

    def learn(self, steps=1000):
        """
        Linear approximation Q-Learning

        Parameters
        __________
        steps : int (default 1000)
            Number of steps that algorithm is repeated
        """

        # Initial state
        obs, amb_stat_dict = self.env.reset()
        state = self.state_encoding(obs, amb_stat_dict)

        for i in range(steps):

            # Choose action
            action, value = self.action_epsilon_greedy(state)

            # Take action
            obs, rew, done, amb_stat_dict, info = self.env.step(action)

            # Next state and estimated value
            next_state = self.state_encoding(obs, amb_stat_dict)
            next_action, next_value = self.policy_action(next_state)

            # Update matrix
            self.feature_vector[action] += self.alpha * (rew + self.gamma * next_value - value) * state
            self.losses.append((rew + self.gamma * next_value - value) ** 2)

            state = next_state

    def plot_loss(self, save=None):
        """
        Plots the graphic corresponding to the evolution of the loss

        Parameters
        __________
        save : str or None
            if None (default) the plot is shown in the screen, otherwise the file is saved and save contains additional
            information to include in the .png file name
        """

        plt.title('Loss training LQL')
        smooth_loss = [sum(self.losses[100*j:101*j])/100 for j in range(1000)]
        plt.plot(smooth_loss)
        #plt.plot(self.losses)
        plt.yscale('log')
        if save is None:
            plt.show()
        else:
            plt.savefig('Training_plots/lql_loss_' + save + '.png')
        plt.close()


def train_model(best=3086.5581, n_trials=10, steps=500000):
    """
    To train Linear Q-Learning and save the best model

    Parameters
    __________
    best : float
        average reward best model obtained so far
    n_trials : int
        number of trials to compare models
    steps : int
        number of steps to train the model
    """

    for i in range(n_trials):  # Choose the best model of n_trials trials

        print('Trial', i)

        # Initialize environment and Linear Q-Learning
        env = Environment()
        env.reset()
        qlearning = Qlearning(env)

        # Train the model
        qlearning.learn(steps)

        # Plot loss
        qlearning.plot_loss(datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Evaluate learned policy
        np.random.seed(1)  # Same random seed for fair comparisons
        seeds = np.random.randint(0, Environment().incidents_generator.num_incidents - 101, 5)  # Starting points
        result = test(env, seeds, qlearning)  # Test

        with open(str(i) + '.txt', 'w') as f:
            f.write(str(i) + ' ' + str(result))
        if result > best:
            best = result
            with open('Models/LQL_online.pickle', 'wb') as handle:
                pickle.dump(qlearning, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(best)


def test(env, seeds, qlearning=None, ep_length=100):
    """
    Test the learned policy

    Parameters
    __________
    env : environment.CustomEnv()
        environment
    seeds : list (of int)
        starting indices incident generator to calculate rewards
    qlearning : Qlearning() or None (Default)
        Linear Q-Learning model to evaluate, if None it uses the stored one

    Return
    ______
    float
        average reward with all seeds
    """

    if qlearning is None:
        with open('Models/LQL_online.pickle', 'rb') as handle:
            qlearning = pickle.load(handle)
    all_rewards = []

    for init in seeds:
        env.reset()
        env.incidents_generator.idx = init
        obs, _, _, amb_stat, _ = env.step(18)
        total_reward_lql = 0
        with open('Results/LinearQL_' + str(init) + '.txt', 'w') as f:
            for _ in range(ep_length):
                f.write(env.render())
                state = qlearning.state_encoding(obs, amb_stat)
                action, _ = qlearning.policy_action(state)
                obs, reward, _, amb_stat, _ = env.step(action)
                total_reward_lql += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_lql, 4)) + '\n')

        print('Reward LQL (' + str(init) + '): ' + str(total_reward_lql))
        all_rewards.append(total_reward_lql)

    average = np.mean(np.array(all_rewards))
    print('Average reward LQL:', average)
    print('__________________________________________________')
    return average

#train_model()

