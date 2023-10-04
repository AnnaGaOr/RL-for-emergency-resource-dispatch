"""
Soft Actor Critic (SAC)
"""

import tensorflow as tf
import keras
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

from environment import *

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

NUM_STATIONS = 18
SHAPE_STATE = 7 * NUM_STATIONS
SIZE_BUFFER = 250
eps = np.finfo(np.float32).eps.item()  # Smallest number > 0


class SoftActorCritic:
    """
    Class to train and test the algorithm SAC

    ...

    Attributes
    __________
    env : environment.CustomEnv()
        environment
    initial_amb : list (of int, len NUM_STATIONS)
        initial number of ambulances per station
    gamma : float (0 < gamma < 1)
        discount factor for the long term reward (default = 0.99)
    tau : float (0 < tau < 1)
        magnitude ot the target updated in each step (default = 0.01)
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
    learn(steps=1000)
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
        self.gamma = 0.9
        self.tau = 0.01

        # Actor and Critics
        self.actor = Actor(actor)
        self.critic1 = Critic(critic1)
        self.critic2 = Critic(critic2)
        self.critic_target1 = Critic(critic1)
        self.critic_target1.model.set_weights(self.critic1.model.get_weights())
        self.critic_target2 = Critic(critic2)
        self.critic_target2.model.set_weights(self.critic2.model.get_weights())
        self.alpha = 1
        self.losses_alpha = []

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

    def learn(self, steps=1000):
        """
        SAC algorithm to learn a policy

        Parameters
        __________
        steps : int (default 1000)
            Number of steps that algorithm is repeated
        """

        for i in range(steps):
            print(i)
            # Replay buffer
            states = np.zeros(SIZE_BUFFER, dtype=object)
            actions = np.zeros(SIZE_BUFFER, dtype=object)
            rewards = np.zeros(SIZE_BUFFER, dtype=object)
            next_states = np.zeros(SIZE_BUFFER, dtype=object)

            # First observation
            obs, amb_stat = self.env.reset()
            next_state = self.state_encoding(obs, amb_stat)

            for j in range(SIZE_BUFFER):

                # Current state
                state = next_state
                state1 = tf.convert_to_tensor(state)
                state1 = tf.expand_dims(state1, 0)
                # Sample action
                action = self.actor.sample_actions(state1)[0]
                # If no available ambulances in station choose an ambulance from another
                if state[action] == 0:
                    probs, _ = self.actor.probabilities(state1)
                    order = np.argsort(probs[0])
                    k = NUM_STATIONS
                    found = False
                    while not found:
                        k -= 1
                        if state[order[k]] > 0:
                            action = order[k]
                            found = True
                # Do action -> next state, reward
                obs, reward, _, amb_stat, _ = self.env.step(action)
                next_state = self.state_encoding(obs, amb_stat)

                # Store s,a,r,s' in the buffer
                states[j] = state
                actions[j] = action
                rewards[j] = reward
                next_states[j] = next_state

            # Update parameters

            # Critics
            # Q_t(s_{t+1},a_{t+1})
            q_t1 = self.critic_target1.model(tf.stack(next_states))
            q_t2 = self.critic_target2.model(tf.stack(next_states))
            q_t = tf.minimum(q_t1, q_t2)
            # pi(a_{t+1}|s_{t+1}), log(pi(a_{t+1}|s_{t+1}))
            probs, logs = self.actor.probabilities(tf.stack(next_states))
            approx = (rewards + self.gamma * (tf.reduce_sum(probs * (q_t - self.alpha * logs), axis=1)))
            approx = (approx - tf.reduce_mean(approx))/ tf.math.reduce_std(approx)
            self.critic1.train(states, actions, approx)
            self.critic2.train(states, actions, approx)

            # Actor
            self.actor.train(states, self.critic1, self.critic2, self.alpha)

            # alpha
            self.update_alpha(states)

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
        fig.suptitle('Evolution losses SAC')
        self.actor.plot_loss(axs[0, 0])
        self.critic1.plot_loss(axs[0, 1], '1')
        self.critic2.plot_loss(axs[1, 0], '2')
        self.plot_loss_alpha(axs[1, 1])
        if save is None:
            plt.show()
        else:
            plt.savefig('Training_plots/sac_loss_' + save + '.png')
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

        # Network
        if model is None:
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
        logs = tf.math.log(probs + eps)
        return probs, logs

    def sample_actions(self, states):
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

        probs = self.model(states)
        actions = [np.random.choice(NUM_STATIONS, p=np.squeeze(dist)) for dist in probs]
        return actions

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
            self.model.add(Dense(512, input_shape=(SHAPE_STATE,), activation='relu'))
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


def train_model(best=2020.8010, n_trials=5, steps=5000):
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

        # Initialize environment and SAC
        env = Environment()
        env.reset()
        sac = SoftActorCritic(env)

        # Train the model
        sac.learn(steps)

        # Plot loss
        sac.plot_loss(datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Evaluate learned policy
        np.random.seed(0)
        seeds = np.random.randint(0, env.incidents_generator.num_incidents-101, 5)
        result = test(env, seeds, sac)

        # Store best model
        if result > best:
            best = result
            sac.actor.model.save('Models/SAC_actor.h5')
            sac.critic1.model.save('Models/SAC_critic1.h5')
            sac.critic2.model.save('Models/SAC_critic2.h5')

    print('Average reward best model:', best)


def test(env, seeds, sac=None, ep_length=100):
    """
    Test the learned policy

    Parameters
    __________
    env : environment.CustomEnv()
        environment
    seeds : list (of int)
        starting indices incident generator to calculate rewars
    sac : SAC() or None (Default)
        SAC model to evaluate, if None it uses the stored one

    Return
    ______
    float
        average reward with all seeds
    """

    # If model is None load it
    if sac is None:
        actor = keras.models.load_model('Models/SAC_actor.h5')
        critic1 = keras.models.load_model('Models/SAC_critic1.h5')
        critic2 = keras.models.load_model('Models/SAC_critic2.h5')
        sac = SoftActorCritic(env, actor, critic1, critic2)

    all_rewards = []

    for init in seeds:
        env.reset()
        env.incidents_generator.idx = init
        obs, _, _, amb_stat, _ = env.step(18)
        total_reward_sac = 0
        with open('Results/sac_' + str(init) + '.txt', 'w') as f:
            for _ in range(ep_length):
                f.write(env.render())
                state = sac.state_encoding(obs, amb_stat)
                action = sac.actor.sample_actions(tf.expand_dims(state, 0))[0]
                if amb_stat[action]['num'] == 0:
                    state1 = tf.convert_to_tensor(state)
                    state1 = tf.expand_dims(state1, 0)
                    probs, _ = sac.actor.probabilities(state1)
                    order = np.argsort(probs[0])
                    k = NUM_STATIONS
                    found = False
                    while not found:
                        k -= 1
                        if state[order[k]] > 0:
                            action = order[k]
                            found = True
                obs, reward, _, amb_stat, _ = env.step(action)
                total_reward_sac += reward
                f.write('Action: ' + str(action) + '\tReward: ' + str(round(reward, 4)) + '\tAccumulated reward: ' + str(round(total_reward_sac, 4)) + '\n')

        print('Reward SAC (' + str(init) + '): ' + str(total_reward_sac))
        all_rewards.append(total_reward_sac)

    average = np.mean(np.array(all_rewards))
    print('Average reward SAC:', average)
    print('__________________________________________________')
    return average


#train_model()
