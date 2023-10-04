"""
Safe Model-Based Policy Optimization (SMBPO)
"""

import tensorflow as tf
import keras
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from datetime import datetime, timedelta
from copy import deepcopy

from environment import Environment
from incidents_generator import IncidentsGenerator

NUM_STATIONS = 18
SHAPE_STATE = 7 * NUM_STATIONS
SIZE_BUFFER_1 = 500
EPS = np.finfo(np.float32).eps.item()  # Smallest number > 0
N_ACTOR = 2
N_ROLLOUT = 5
MINI_BATCH = 250
GAMMA = 0.99
TAU = 0.01
HORIZON = 21
COST = (GAMMA ** - HORIZON) * 110
SIZE_BUFFER_2 = N_ROLLOUT * HORIZON


class SMBPO:
    """
    Class to train and test the algorithm SMBPO

    ...

    Attributes
    __________
    env : environment.CustomEnv()
        environment
    initial_amb : list (of int, len NUM_STATIONS)
        initial number of ambulances per station
    horizon : int
        parameter horizon (default = HORIZON)
    gamma : float (0 < gamma < 1)
        discount factor for the long term reward (default = GAMMA)
    tau : float (0 < tau < 1)
        magnitude ot the target updated in each step (default = TAU)
    buffer1 : collections.deque()
        replace buffer with experiences extracted from the environment
    buffer2 : collections.deque()
        replace buffer with experiences extracted from dynamics
    dynamics : Dynamics()
        class to predict a next possible accident
    actor : Actor()
        actor
    critic1 : Critic()
        critic 1
    critic2 : Critic()
        critic 2
    critic_target1 : Critic()
        target critic 1
    critic_target2 : Critic()
        target critic 2
    alpha : float
        temperature parameter
    losses_alpha : list
        stores the losses of alpha

    Methods
    _______
    state_encoding(obs, amb_stat_dict)
        Encodes the state to be entered in the NN
    update_alpha(states)
        Updates alpha
    learn(steps=1000, length=100)
        SAC algorithm to learn a policy
    plot_loss(save=None)
        Save plot loss
    plot_loss_alpha(axs)
        Plot of alpha
    """

    def __init__(self, env, actor=None, critic1=None, critic2=None):
        """
        Parameters
        __________
        env : environment.CustomEnv()
            environment
        actor : keras.Sequential() or None (default)
            model for the actor, if it is given it initializes the NN as actor, otherwise it initializes a Network to be
            trained
        critic1 : keras.Sequential() or None (default)
            model for the critic1, if it is given it initializes the NN as critic1, otherwise it initializes a Network
            to be trained
        critic2 : keras.Sequential() or None (default)
            model for the critic2, if it is given it initializes the NN as critic2, otherwise it initializes a Network
            to be trained
        """

        # Environment
        self.env = env
        self.env.reset()

        # Initial ambulances per station
        amb_dict = self.env.ambu_in_stations
        self.initial_amb = []
        for j in range(NUM_STATIONS):
            self.initial_amb.append(amb_dict[j]['num'])

        # Parameters
        self.horizon = HORIZON
        self.gamma = GAMMA
        self.tau = TAU

        # Replace buffers
        self.buffer1 = deque(maxlen=SIZE_BUFFER_1)
        self.buffer2 = deque(maxlen=SIZE_BUFFER_2)

        # Dynamics
        self.dynamics = Dynamics()

        # Actor
        self.actor = Actor(actor)

        # Critics
        self.critic1 = Critic(critic1)
        self.critic2 = Critic(critic2)
        self.critic_target1 = Critic(critic1)
        self.critic_target2 = Critic(critic2)
        self.critic_target1.model.set_weights(self.critic1.model.get_weights())
        self.critic_target2.model.set_weights(self.critic2.model.get_weights())

        # alpha, loss alpha
        self.alpha = 1
        self.losses_alpha = []

        # Initial random data
        actions = np.random.randint(0, NUM_STATIONS, SIZE_BUFFER_1)  # Sample random actions
        next_obs, amb_stat = self.env.reset()
        next_state = self.state_encoding(next_obs, amb_stat)
        next_state = tf.convert_to_tensor(next_state)
        for action in actions:
            state = next_state  # State
            obs = next_obs
            if state[action] > 0:  # Check if action is possible
                next_obs, reward, _, amb_stat, _ = self.env.step(action)  # Do action, observe reward and next state
                next_state = self.state_encoding(next_obs, amb_stat)
                next_state = tf.convert_to_tensor(next_state)
                busy_amb = deepcopy(self.env.busy_ambulances)
                if obs[1] == 0 and reward < 0:  # Action not safe  ->  reward=-COST
                    self.buffer1.append([np.array(state), action, - COST, np.array(state), obs[0], obs[1], np.array(obs[3]), busy_amb])
                else:  # Safe action
                    self.buffer1.append([np.array(state), action, reward, np.array(next_state), next_obs[0], next_obs[1], np.array(next_obs[3]), busy_amb])

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

        state = np.zeros(SHAPE_STATE)
        state[:NUM_STATIONS] = amb_stat
        if priority == 0:
            state[NUM_STATIONS:2 * NUM_STATIONS] = amb_stat
            state[2 * NUM_STATIONS:3 * NUM_STATIONS] = np.array(obs[3]) / 60
        elif priority == 1:
            state[3 * NUM_STATIONS:4 * NUM_STATIONS] = amb_stat
            state[4 * NUM_STATIONS:5 * NUM_STATIONS] = np.array(obs[3]) / 60
        else:
            state[5 * NUM_STATIONS:6 * NUM_STATIONS] = amb_stat
            state[6 * NUM_STATIONS:] = np.array(obs[3]) / 60

        return state

    def update_alpha(self, states):
        """
        Update alpha using the loss function J(alpha) = E_{a~pi} [- alpha * log pi(a|s) - alpha * mean_entropy]

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            minibatch of states to estimate the loss to update alpha
        """

        with tf.GradientTape() as tape:
            mean_entropy = 0.0001
            # pi(a_t|s_t), log(pi(a_t|s_t))
            probs, logs = self.actor.probabilities(tf.stack(states))

            entropy = tf.reduce_sum(probs * logs, axis=1)

            grad_loss = -tf.reduce_mean(entropy + mean_entropy)

            self.alpha -= 0.001 * grad_loss
            self.losses_alpha.append(self.alpha * grad_loss)

    def learn(self, steps=1000, length=100):
        """
        SMBPO algorithm to learn a policy

        Parameters
        __________
        steps : int (default 1000)
            Number of steps that algorithm is repeated
        length : int (default 100)
            Length of an episode
        """

        for i in range(steps):

            # Collect episode
            next_obs, amb_stat = self.env.reset()
            next_state = self.state_encoding(next_obs, amb_stat)
            next_state = tf.convert_to_tensor(next_state)
            total_reward = 0
            for j in range(length):
                # Current state
                state = next_state
                obs = next_obs
                # Sample action
                action = self.actor.sample_actions(tf.expand_dims(state, 0))[0]
                # Do action -> next state, reward
                next_obs, reward, _, amb_stat, _ = self.env.step(action)
                total_reward += reward
                next_state = self.state_encoding(next_obs, amb_stat)
                next_state = tf.convert_to_tensor(next_state)
                busy_amb = deepcopy(self.env.busy_ambulances)
                # Store s,a,r,s' in the buffer
                if obs[1] == 0 and reward < 0:
                    self.buffer1.append([np.array(state), action, - COST, np.array(state), obs[0], obs[1], np.array(obs[3]), busy_amb])
                else:
                    self.buffer1.append([np.array(state), action, reward, np.array(next_state), next_obs[0], next_obs[1], np.array(next_obs[3]), busy_amb])

            # Buffer 2
            sample = np.array(self.buffer1)[np.random.randint(0, len(self.buffer1), N_ROLLOUT)]
            next_states = np.array(list(sample[:, 3]))
            distances = np.concatenate(sample[:, 6]).reshape(N_ROLLOUT, NUM_STATIONS)
            priorities = np.array(sample[:, 5])
            times = np.array(sample[:, 4])
            busy_amb = sample[:, 7]
            for _ in range(HORIZON):
                # States
                states = next_states
                # Actions
                actions = np.array(self.actor.sample_actions(tf.stack(states)))
                # Rewards
                min_dist = np.min(distances, axis=1)
                actions_mask = tf.one_hot(actions, NUM_STATIONS)
                act_dist = tf.reduce_sum(distances * actions_mask, axis=1)
                val = np.array(min_dist / act_dist)
                unsafe = np.where(np.array(priorities == 0) * np.array(val < 0.8))[0]
                safe = np.where(np.array(priorities != 0) + np.array(val >= 0.8))[0]
                rewards = np.zeros(length)
                rewards[safe] = np.array((99 * (priorities[safe] == 0) + 9 * (priorities[safe] == 1) + 1) * (2 * (val[safe] ** 3) - 1))
                rewards[unsafe] = - COST
                # New states
                next_states[safe], distances[safe], priorities[safe], times[safe], busy_amb[safe] = self.dynamics.predict_next_state(states[safe], actions[safe], times[safe], busy_amb[safe], self.initial_amb)
                self.buffer2 += [[np.array(states[i]), actions[i], rewards[i], np.array(next_states[i]), times[i], priorities[i], distances[i]] for i in range(N_ROLLOUT)]

            # Update parameters
            for _ in range(N_ACTOR):

                all_data = list(np.array(self.buffer1)[:, :7]) + list(np.array(self.buffer1)[:, :7])
                sample = np.array(all_data)[np.random.randint(0, len(self.buffer1), MINI_BATCH)]

                # Critics
                # V(s')
                # Q_t(s_{t+1},a_{t+1})
                q_t1 = self.critic_target1.model(tf.stack(sample[:, 3]))
                q_t2 = self.critic_target2.model(tf.stack(sample[:, 3]))
                q_t = tf.minimum(q_t1, q_t2)

                # pi(a_{t+1}|s_{t+1}), log(pi(a_{t+1}|s_{t+1}))
                probs, logs = self.actor.probabilities(tf.stack(sample[:, 3]))
                v = np.array(tf.reduce_sum(probs * (q_t - self.alpha * logs), axis=1))

                v = tf.convert_to_tensor(v)
                # Approximation
                approx = (sample[:, 2] + GAMMA * v)
                approx = (approx - tf.reduce_mean(approx))/ tf.math.reduce_std(approx)
                self.critic1.train(sample[:, 0], sample[:, 1], approx)
                self.critic2.train(sample[:, 0], sample[:, 1], approx)

                # Actor
                self.actor.train(sample[:, 0], self.critic1, self.critic2, self.alpha)

                # alpha
                self.update_alpha(sample[:, 0])

                # Critic targets
                critic1_weights = np.array(self.critic1.model.get_weights(), dtype=object)
                target1_weights = np.array(self.critic_target1.model.get_weights(), dtype=object)
                updated_weights1 = self.tau*critic1_weights + (1-self.tau)*target1_weights
                self.critic_target1.model.set_weights(updated_weights1)
                critic2_weights = np.array(self.critic2.model.get_weights(), dtype=object)
                target2_weights = np.array(self.critic_target2.model.get_weights(), dtype=object)
                updated_weights2 = self.tau*critic2_weights + (1-self.tau)*target2_weights
                self.critic_target2.model.set_weights(updated_weights2)

    def plot_loss(self, save=None):
        """
        Plots the graphic corresponding to the evolution of all losses: actor, both critics and alpha

        Parameters
        __________
        save : str or None
            if None (default) the plot is shown in the screen, otherwise the file is saved and save contains additional
            information to include in the .png file name

        """

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle('Evolution losses SMBPO')
        self.actor.plot_loss(axs[0, 0])
        self.critic1.plot_loss(axs[0, 1], '1')
        self.critic2.plot_loss(axs[1, 0], '2')
        self.plot_loss_alpha(axs[1, 1])
        if save is None:
            plt.show()
        else:
            plt.savefig('Training_plots/smbpo_loss_' + save + '.png')
        plt.close()

    def plot_loss_alpha(self, axs):
        """
        Plots the loss of alpha

        Parameters
        __________
        axs : Axes or array of Axes
            place to do the plot
        """

        axs.plot(np.abs(np.array(self.losses_alpha)))
        axs.set_title('Alpha')
        axs.set_yscale('log')


class Dynamics:
    """
    Model of the environment, this class predicts a possible new state
    As I know how the environment is, I will explicitly define the dynamics (without any training)

    ...

    Attributes
    __________
    ig : IncidentGenerator()
        class to access the dataset
    num_accidents : int
        total number of accidents in the dataset
    num : int
        number of new states to predict

    Methods
    _______
    predict_next_state(states, actions, times, busy_amb, init_amb)
        predicts a possible next state
    accident_ends(ini_times):
        time while the ambulance is busy
    next_accident(prev_times):
        time for the next accident
    coordinates()
        predict the location of the next incident
    """

    def __init__(self):

        self.ig = IncidentsGenerator()
        self.num_accidents = self.ig.num_incidents

        self.num = None  # Number new states to predict

    def predict_next_state(self, states, actions, times, busy_amb, init_amb):
        """
        Predicts a possible next state

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            states
        actions : list of int
            actions done in the state
        times : numpy.array() of datatime.datatime()
            times of the state
        busy_amb : list of dict
            list of a dictionary of busy ambulances per state
        initial_amb : list (of int, len NUM_STATIONS)
            initial number of ambulances per station


        Returns
        _______
        numpy.array() of numpy.array() with shape=(SHAPE_STATE,)
            for each state and action, the next states
        numpy.array() of list of floats
            distances to each station
        numpy.array() of int in {0,1,2}
            priority level of the new emergency
        numpy.array() of datatime.datatime()
            times of the next emergencies, when the next accident will occur
        list of list
            List of busy ambulances, contains tuples with the station of prominence of the ambulance and the time when it
        will be available again.
        """

        self.num = len(states)  # Number of new states to predict

        amb_stat = np.array(states)[:, :18] * np.tile(np.array(init_amb), (self.num, 1))  # available ambulances (before action)

        # Add ambulance to busy ambulances
        end_times = self.accident_ends(times)
        for i in range(self.num):
            busy_amb[i].append((actions[i], end_times[i]))
        actions_mask = tf.one_hot(actions, NUM_STATIONS).numpy()
        amb_stat -= actions_mask

        # Next accidents
        new_times = self.next_accident(times)  # Next times
        priorities = np.random.choice([0, 1, 2], p=(0.2718828868180936, 0.4242926130848379, 0.3038245000970685), size=self.num)  # Next priorities
        coordinates, distances = self.coordinates()
        obs = new_times, priorities, coordinates, distances

        # Update state of ambulances for the new times
        for i in range(self.num):
            indexes = [j for j in range(len(busy_amb[i]))]
            for j in sorted(indexes, reverse=True):
                if busy_amb[i][j][1] < new_times[i]:
                    amb_stat[i][busy_amb[i][j][0]] += 1
                    busy_amb[i].pop(j)
        amb_stat = amb_stat/init_amb

        # New states
        next_states = np.zeros((self.num, SHAPE_STATE))
        next_states[:, :NUM_STATIONS] = amb_stat
        information = np.concatenate((amb_stat, np.array(distances) / 60), axis=1)
        aux_inf = np.tile(information, 3)
        aux_inf[(priorities != 0), :2*NUM_STATIONS] = 0
        aux_inf[(priorities != 1), 2*NUM_STATIONS:4*NUM_STATIONS] = 0
        aux_inf[(priorities != 2), 4*NUM_STATIONS:] = 0
        next_states[:, NUM_STATIONS:] = aux_inf

        return next_states, distances, priorities, new_times, busy_amb

    def accident_ends(self, ini_times):
        """
        Generate accident time following a gamma distribution, params are the same as in the environment

        Parameters
        __________
        ini_time : numpy.array() of datatime.datatime()
            initial times of the emergencies

        Return
        ______
        numpy.array() of datatime.datatime()
            end times of the emergencies, when the ambulance is available again
        """

        accident_times = np.random.gamma(3.9110986587617522, 1109.3708330737102, self.num)
        accident_times = np.array([timedelta(seconds=time) for time in accident_times])
        end_time = ini_times + accident_times

        return end_time

    def next_accident(self, prev_times):
        """
        Generate the time for the new accident assuming that the accidents follow a Poisson process

        Parameters
        __________
        prev_time : numpy.array() of datatime.datatime()
            times of the previous emergencies

        Returns
        _______
        numpy.array() of datatime.datatime()
            times of the next emergencies, when the next accident will occur
        """

        lam = 0.0016782111559257636  # Estimated parameter

        # Generate sample Poisson process
        uniform = np.random.uniform(0, 1, self.num)
        durations = - np.log(uniform) / lam
        durations = np.array([timedelta(seconds=sec) for sec in durations])

        next_times = prev_times + durations
        return next_times

    def coordinates(self):
        """
        To predict the location of the next incident

        Returns
        _______
        list of tuples
            coordinates for the position of the new emegencies
        numpy.array()
            distance to each station
        """

        # For the coordinates, to ensure that it is a possible location, we take randomly among the ones in the data
        indices = np.random.randint(0, self.num_accidents, self.num)
        coordinates = []
        distances = []
        for idx in indices:
            self.ig.idx = idx
            _, _, coord, dist = self.ig.new_emergency()
            coordinates.append(coord)
            distances.append(dist)
        distances = np.array(distances).reshape((self.num, NUM_STATIONS))

        return coordinates, distances


class Actor:
    """
    Actor, network to learn the policy

    ...

    Attributes
    __________
    model : keras.Sequential()
        Network for the actor
    optimizer : keras.optimizers()
        Optimizer
    losses : list
        To store losses

    Methods
    _______
    probabilities(states)
        probabilities of actions and its logarithms
    sample_actions(states)
        Samples action in a state according to the probabilities
    train(states, critic1, critic2, alpha)
        Updates the parameters of the model
    plot_loss(axs)
        Plot loss
    """

    def __init__(self, model=None):
        """
        model : keras.Sequential() or None (default)
            model for the actor, if it is given it initializes the NN as model, otherwise it initializes a Network to be
            trained
        """

        if model is None:
            # Network
            init = keras.initializers.RandomNormal(stddev=0.001)
            self.model = keras.Sequential()
            self.model.add(Dense(512, input_shape=(SHAPE_STATE,), activation='relu', kernel_initializer=init))
            self.model.add(Dense(512, activation='relu', kernel_initializer=init))
            self.model.add(Dense(NUM_STATIONS, activation='softmax', kernel_initializer=init))
        else:
            self.model = model

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # To store losses
        self.losses = []

    def probabilities(self, states):
        """
        Probability and log(probability) od each action in each state

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            states

        Returns
        _______
        tensorflow.Tensor()
            Probability of each action in each state
        tensorflow.Tensor()
            Logarithm of the probability
        """

        probs = self.model(states)
        logs = tf.math.log(probs + EPS)
        return probs, logs

    def sample_actions(self, state):
        """
        Samples action in a state according to the probabilities

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            states

        Returns
        _______
        list
            actions
        """

        probs = self.model(state) + EPS
        probs = (np.array(state[:, :NUM_STATIONS]) > 0) * probs
        probs = tf.linalg.normalize(probs, ord=1, axis=1)[0]

        action = [np.random.choice(NUM_STATIONS, p=np.squeeze(dist)) for dist in probs]

        return action

    def train(self, states, critic1, critic2, alpha):
        """
        Updates the parameters of the model

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            minibatch of states to estimate the loss to update the actor
        critic1 : Critic()
            critic 1
        critic2 : Critic()
            critic 2
        alpha : float
            temperature
        """

        with tf.GradientTape() as tape:
            # pi(a_t,s_t), log(pi(a_t,s_t))
            pi, log = self.probabilities(tf.stack(states))
            # Q(s_t,a_t)
            q1 = critic1.model(tf.stack(states))
            q2 = critic2.model(tf.stack(states))
            q = tf.minimum(q1, q2)

            loss = tf.reduce_mean(tf.reduce_sum(pi * (alpha * log - q), axis=1))
            self.losses.append(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def plot_loss(self, axs):
        """
        Plots the loss of the actor

        Parameters
        __________
        axs : Axes or array of Axes
            place to do the plot
        """

        axs.plot(np.array(self.losses))
        axs.set_title('Actor')


class Critic:
    """
    Critic, network to evaluate the policy

    ...

    Attributes
    __________
    model : keras.Sequential()
        Network for the actor
    optimizer : keras.optimizers()
        Optimizer
    losses : list
        To store losses

    Methods
    _______
    train(states, actions, approx)
        Updates the parameters of the model
    plot_loss(axs, num)
        Plot loss
    """

    def __init__(self, model=None):
        """
        critic : keras.Sequential() or None (default)
            model for the critic, if it is given it initializes the NN as model, otherwise it initializes a Network to
            be trained
        """

        # Network
        if model is None:
            self.model = keras.Sequential()
            self.model.add(Dense(256, input_shape=(SHAPE_STATE,), activation='relu'))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(NUM_STATIONS))
        else:
            self.model = model

        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # To store losses
        self.losses = []

    def train(self, states, actions, approx):
        """
        Updates the parameters of the model

        Parameters
        __________
        states : numpy.array() or list of (numpy.array(), shape=(SHAPE_STATE,))
            minibatch of states to estimate the loss to update the critic
        actions : numpy.array()
            actions done by the actor in the states
        approx : tensorflow.Tensor()
            estimation of the Q-value (r + gamma * E[V_t(a')]) to calculate the loss as the MSE
        """

        with tf.GradientTape() as tape:
            # Q(s_t,a_t)
            q = self.model(tf.stack(states))
            actions = tf.one_hot(actions, NUM_STATIONS)
            q = tf.reduce_sum(tf.multiply(actions, q), axis=1)
            approx = tf.cast(approx, dtype=q.dtype)
            loss = tf.reduce_mean((q - approx) ** 2)
            self.losses.append(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def plot_loss(self, axs, num):
        """
        Plots the loss of the critic

        Parameters
        __________
        axs : Axes or array of Axes
            place to do the plot
        num: str
            Name of the critic
        """

        axs.plot(self.losses)
        axs.set_title('Critic ' + num)
        axs.set_yscale('log')


def train_model(best=1425.2952, n_trials=5, steps=1000):
    """
    To train SAC and save the best model

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

        # Initialize environment and SMBPO
        env = Environment()
        env.reset()
        smbpo = SMBPO(env)

        # Train the model
        smbpo.learn(steps)

        # Plot loss
        smbpo.plot_loss(datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Evaluate learned policy
        np.random.seed(0)
        seeds = np.random.randint(0, env.incidents_generator.num_incidents-101, 5)
        result = test(env, seeds, smbpo)

        # Store best model
        if result > best:
            best = result
            smbpo.actor.model.save('Models/SMBPO_actor.h5')
            smbpo.critic1.model.save('Models/SMBPO_critic1.h5')
            smbpo.critic2.model.save('Models/SMBPO_critic2.h5')

    print('Average reward best model:', best)


def test(env, seeds, smbpo=None, ep_length=100):
    """
    Test the learned policy

    Parameters
    __________
    env : environment.CustomEnv()
        environment
    seeds : list (of int)
        starting indices incident generator to calculate rewars
    smbpo : SMBPO() or None (Default)
        SMBPO model to evaluate, if None it uses the stored one

    Return
    ______
    float
        average reward with all seeds
    """

    # If model is None load it
    if smbpo is None:
        actor = keras.models.load_model('Models/SMBPO_actor.h5')
        critic1 = keras.models.load_model('Models/SMBPO_critic1.h5')
        critic2 = keras.models.load_model('Models/SMBPO_critic2.h5')
        smbpo = SMBPO(env, actor, critic1, critic2)

    all_rewards = []

    for init in seeds:
        env.reset()
        env.incidents_generator.idx = init
        obs, _, _, amb_stat, _ = env.step(18)
        total_reward_smbpo = 0
        with open('Results/smbpo_' + str(init) + '.txt', 'w') as f:
            for _ in range(ep_length):
                f.write(env.render())
                state = smbpo.state_encoding(obs, amb_stat)
                action = smbpo.actor.sample_actions(tf.expand_dims(state, 0))[0]
                obs, reward, _, amb_stat, _ = env.step(action)
                total_reward_smbpo += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_smbpo, 4)) + '\n')

        print('Reward SMBPO (' + str(init) + '): ' + str(total_reward_smbpo))
        all_rewards.append(total_reward_smbpo)

    average = np.mean(np.array(all_rewards))
    print('Average reward SMBPO:', average)
    print('__________________________________________________')
    return average


# train_model(steps=1000, n_trials=10)
