"""
· Deep Q-Learning (DQN) Q-Learning approximating Q with a Neural Network
In DQN the loss function is Q(s,a) - (r - gamma * max(Q_t(s',a')))
· Double DQN (Double-DQN) DQN changing the loss function to avoid overestimation of Q values
IN Double-DQN the loss function is Q(s,a) - (r - gamma * Q_t(s', argmax_{a'}(Q(s',a'))))
"""

import keras
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt
from datetime import datetime

from environment import *

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

NUM_STATIONS = 18
SHAPE_STATE = 7 * NUM_STATIONS
MINI_BATCH_SIZE = 64
MIN_SIZE_BUFFER = 4*MINI_BATCH_SIZE


class DQN:
    """
    Class to train and test the algorithm DQN or Double-DQN

    ...

    Attributes
    __________
    epsilon : float (0 < epsilon < 1)
        probability of a random action in epsilon-greedy policy (0.01 + 0.99 * np.exp(-0.01 * step))
    gamma : float (0 < gamma < 1)
        discount factor for the long term reward (default = 0.9)
    tau : float (0 < tau < 1)
        magnitude ot the target updated in each step (default = 0.01)
    env : environment.CustomEnv()
        environment
    initial_amb : list (of int, len NUM_STATIONS)
        initial number of ambulances per station
    double : bool
        if False it executes DQN, if True Double-DQN
    model : keras.Sequential()
        NN approximating Q
    target : keras.Sequential()
        target network
    optimizer : keras.optimizers()
        optimizer (default keras.optimizers.Adam(learning_rate=0.005))
    buffer : deque()
        replay buffer to store experiences
    losses : list
        list of losses to plot the evolution

    Methods
    _______
    generate_model(prev_model=None)
        Initializes the neural network to approximate Q
    action_epsilon_greedy(state)
        Samples action following the epsilon greedy policy
    random_action(state)
        Samples a random action
    policy_action(state)
        Action according to the learned policy
    state_encoding(obs, amb_stat_dict)
        Encodes the state to be entered in the NN
    learn(steps=1000)
        DQN algorithm to learn a policy
    train()
        Update model
    plot_loss(save=None)
        Save plot loss
    """

    def __init__(self,  env, double=False, model=None):
        """
        Parameters
        __________
        env : environment.CustomEnv()
            environment
        double : bool (Default False)
            If False DQN, if True Double-DQN
        model : keras.Sequential() or None (Default)
            If model is given it initializes the NN as model, otherwise it initializes a  Network to be trained
        """

        # Parameters
        self.epsilon = None
        self.gamma = 0.9
        self.tau = 0.01

        # Environment and initial ambulances in stations
        self.env = env
        amb_dict = self.env.ambu_in_stations
        self.initial_amb = []
        for j in range(NUM_STATIONS):
            self.initial_amb.append(amb_dict[j]['num'])

        # Algorithm (DQN or Double-DQN)
        self.double = double

        # Initialize NNs
        self.model = self.generate_model(model)
        self.target = self.generate_model(model)
        self.target.set_weights(self.model.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # Replace buffer
        self.buffer = deque(maxlen=MIN_SIZE_BUFFER)  # buffer

        self.losses = []  # to store losses

    def generate_model(self, prev_model=None):
        """
        Initializes the neural network to approximate Q

        Parameters
        _________
        prev_model : keras.Sequential() or None (Default)
            If prev_model is given it initializes the NN as prev_model, otherwise it initializes a Network to be
            trained

        Returns
        _______
        keras.Sequential()
            NN to approximate Q
        """

        if prev_model is None:
            init = tf.keras.initializers.HeUniform()
            model = keras.Sequential()
            model.add(Dense(512, input_shape=(SHAPE_STATE,), activation='relu', kernel_initializer=init))
            model.add(Dense(512, activation='relu', kernel_initializer=init))
            model.add(Dense(512, activation='relu', kernel_initializer=init))
            model.add(Dense(NUM_STATIONS, activation='linear', kernel_initializer=init))

        else:
            model = prev_model

        return model

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
            action = self.random_action(state)

        else:
            action = self.policy_action(state)

        return action

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

        # Sample random actions until it finds a possible one
        found = False
        while not found:
            act = np.random.randint(0, NUM_STATIONS + 1)
            if act == NUM_STATIONS or state[act] > 0:
                found = True

        return act

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

        rewards_actions = self.model(tf.expand_dims(state, 0))[0]  # Q values

        # Choose the possible action with higher Q value
        order = np.argsort(rewards_actions)

        found = False
        i = NUM_STATIONS - 1
        while not found:
            act = order[i]
            if act == NUM_STATIONS or state[act] > 0:
                found = True
            i -= 1

        return act

    def state_encoding(self, obs, amb_stat_dict):
        """
        Encodes the state to be entered in the NN

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

        amb_stat = []
        for j in range(NUM_STATIONS):
            amb_stat.append(amb_stat_dict[j]['num'] / self.initial_amb[j])

        state = np.zeros(7 * NUM_STATIONS)
        state[:NUM_STATIONS] = amb_stat

        if priority == 0:
            state[NUM_STATIONS:2*NUM_STATIONS] = amb_stat
            state[2*NUM_STATIONS:3*NUM_STATIONS] = obs[3]
        elif priority == 1:
            state[3*NUM_STATIONS:4*NUM_STATIONS] = amb_stat
            state[4*NUM_STATIONS:5 * NUM_STATIONS] = obs[3]
        else:
            state[5*NUM_STATIONS:6*NUM_STATIONS] = amb_stat
            state[6*NUM_STATIONS:] = obs[3]

        return state

    def learn(self, steps=1000):
        """
        DQN or DDQN algorithm to learn a policy

        Parameters
        __________
        steps : int (default 1000)
            Number of steps that algorithm is repeated
        """

        # Initial state
        obs, amb_stat_dict = self.env.reset()
        next_state = self.state_encoding(obs, amb_stat_dict)

        for i in range(steps):

            if i % 1000 == 0:
                print(i)

            state = next_state

            # Choose action
            self.epsilon = 0.01 + 0.99 * np.exp(-0.01 * i)
            action = self.action_epsilon_greedy(state)

            # Take action observe reward and next state
            obs, reward, done, amb_stat_dict, info = self.env.step(action)
            next_state = self.state_encoding(obs, amb_stat_dict)
            self.buffer.append([state, action, reward, next_state])

            # Update model every 4 steps
            if i % 4 == 0:
                self.train()

            # Update target model
            model_weights = np.array(self.model.get_weights(), dtype=object)
            target_weights = np.array(self.target.get_weights(), dtype=object)
            updated_weights = self.tau * model_weights + (1 - self.tau) * target_weights
            self.target.set_weights(updated_weights)

    def train(self):
        """
        Calculate the loss and update model
        """

        if len(self.buffer) >= MIN_SIZE_BUFFER:  # enough training instances

            # Sample mini-batch
            mini_batch = random.sample(self.buffer, MINI_BATCH_SIZE)

            # Calculate loss

            with tf.GradientTape() as tape:

                # Approx
                if not self.double:  # DQN (r + gamma * max(Q(s',a')))
                    # r
                    rewards = np.array([instance[2] for instance in mini_batch])
                    # Q_t(s',a')
                    next_states = np.array([instance[3] for instance in mini_batch])
                    next_q_values = self.target(next_states)
                    # max_{a' possible} Q(s',a')
                    not_possible = (next_states[:, :NUM_STATIONS] == 0)
                    next_q_values -= 100000 * not_possible
                    max_q = np.max(next_q_values, axis=1)
                    # approx (r + gamma * max(Q(s',a')))
                    approx = rewards + self.gamma * max_q

                else:  # Double-DQN (r + gamma * Q_t(s', argmax_{a'} Q(s',a')))
                    # r
                    rewards = np.array([instance[2] for instance in mini_batch])
                    # s'
                    next_states = np.array([instance[3] for instance in mini_batch])
                    # Q(s', a')
                    next_q = self.model(next_states)
                    # argmax_{a' possible} Q(s',a')
                    not_possible = (next_states[:, :NUM_STATIONS] == 0)
                    next_q -= 100000 * not_possible
                    act_max = np.argmax(next_q, axis=1)
                    act_max = tf.one_hot(act_max, NUM_STATIONS)
                    # Q_t(s', argmax_{a'} Q(s',a'))
                    next_q_target = self.target(next_states)
                    q_t = tf.reduce_sum(next_q_target * act_max, axis=1)
                    # approx
                    approx = rewards + self.gamma * q_t

                # Q(s,a)
                states = np.array([instance[0] for instance in mini_batch])
                q_values = self.model(states)
                actions = np.array([instance[1] for instance in mini_batch])
                actions = tf.one_hot(actions, NUM_STATIONS)
                q_sa = tf.reduce_sum(q_values * actions, axis=1)

                loss = tf.reduce_mean((approx - q_sa) ** 2)
                self.losses.append(loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def plot_loss(self, save=None):
        """
        Plots the graphic corresponding to the evolution of the loss

        Parameters
        __________
        save : str or None
            if None (default) the plot is shown in the screen, otherwise the file is saved and save contains additional
            information to include in the .png file name
        """

        if not self.double:
            plt.title('Loss training DQN')
        else:
            plt.title('Loss training Double-DQN')
        plt.plot(self.losses)
        plt.yscale('log')
        if save is None:
            plt.show()
        else:
            if not self.double:
                plt.savefig('Training_plots/dqn_loss_' + save + '.png')
            else:
                plt.savefig('Training_plots/ddqn_loss_' + save + '.png')
        plt.close()


def train_model(best, double=False, n_trials=5, steps=100000):
    """
    To train DQN and save the best model

    Parameters
    __________
    best : float
        average reward best model obtained so far
    double : bool (Default False)
        if False it executes DQN, if True Double-DQN
    n_trials : int
        number of trials to compare models
    steps : int
        number of steps to train the model
    """

    for i in range(n_trials):  # Choose the best model of n_trials trials

        # Initialize environment and DQN
        env = Environment()
        env.reset()
        dqn = DQN(env, double)

        # Train the model
        dqn.learn(steps)

        # Plot loss
        dqn.plot_loss(datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Evaluate learned policy
        np.random.seed(1)  # Same random seed for fair comparisons
        seeds = np.random.randint(0, env.incidents_generator.num_incidents-101, 5)  # Starting points
        result = test(env, seeds, double, dqn)  # Test

        # Store best model
        if result > best:
            best = result
            if not double:
                dqn.model.save('Models/DQN_online.h5')
            else:
                dqn.model.save('Models/DDQN_online.h5')

    print('Average reward best model:', best)


def test(env, seeds, double=False, dqn=None, ep_length=100):
    """
    Test the learned policy

    Parameters
    __________
    env : environment.CustomEnv()
        environment
    seeds : list (of int)
        starting indices incident generator to calculate rewards
    double : bool (Default False)
        if False it executes DQN, if True Double-DQN
    dqn : DQN() or None (Default)
        DQN model to evaluate, if None it uses the stored one

    Return
    ______
    float
        average reward with all seeds
    """

    # If dqn not given initialize it with the stored one
    if dqn is None:
        if not double:
            model = keras.models.load_model('Models/DQN_online.h5')
            dqn = DQN(env, double, model)
        else:
            model = keras.models.load_model('Models/DDQN_online.h5')
            dqn = DQN(env, double, model)

    all_rewards = []

    for init in seeds:
        env.reset()
        env.incidents_generator.idx = init
        obs, _, _, amb_stat, _ = env.step(18)
        total_reward_dqn = 0
        if not double:
            alg = 'DQN'
        else:
            alg = 'DDQN'
        with open('Results/' + alg + '_' + str(init) + '.txt', 'w') as f:
            for _ in range(ep_length):  # Calculate reward in an episode of 100 steps
                f.write(env.render())
                state = dqn.state_encoding(obs, amb_stat)
                action = dqn.policy_action(state)
                obs, reward, _, amb_stat, _ = env.step(action)
                total_reward_dqn += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_dqn, 4)) + '\n')

        print('Reward ' + alg + ' (' + str(init) + '): ' + str(total_reward_dqn))
        all_rewards.append(total_reward_dqn)

    average = np.mean(np.array(all_rewards))  # Average reward all episodes
    print('Average reward ' + alg + ': ' + str(round(average, 4)))
    print('__________________________________________________')
    return average


def test_double(env, seeds, double=True, dqn=None, ep_length=100):
    test(env, seeds, double, dqn, ep_length)


# Train Deep Q-Learning
# train_model(best=2334.4437)

# Train Double Deep Q_Learning
# train_model(double=True, best=2345.1283)
