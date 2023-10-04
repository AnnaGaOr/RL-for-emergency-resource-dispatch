"""
Environment
"""

import numpy as np
import gym
import pickle
from datetime import datetime, timedelta

from incidents_generator import IncidentsGenerator


LIST_OF_STATIONS = {
    'Lausanne': {'id': 0, 'lat': 46.523525, 'lng': 6.638332, 'canton': 'VD'},
    'Villars-Sainte-Croix': {'id': 1, 'lat': 46.567019, 'lng': 6.560686, 'canton': 'VD'},   #new
    'Nyon': {'id': 2, 'lat': 46.382640, 'lng':  6.226248, 'canton': 'VD'},    # new
    'Tour-de-Peiz': {'id': 3, 'lat': 46.457491, 'lng': 6.853541, 'canton': 'VD'},
    'Morges': {'id': 4, 'lat': 46.523000, 'lng': 6.501375, 'canton': 'VD'},
    'Yverdons-les-Bains': {'id': 5, 'lat': 46.772147, 'lng': 6.644941, 'canton': 'VD'},
    'Aigle': {'id': 6, 'lat': 46.312054, 'lng': 6.964270, 'canton': 'VD'},
    'Aubonne': {'id': 7, 'lat': 46.491782, 'lng': 6.388352, 'canton': 'VD'},    # new
    'Payerne': {'id': 8, 'lat': 46.819290, 'lng': 6.948423, 'canton': 'VD'},
    'Mézières': {'id': 9, 'lat': 46.593600, 'lng': 6.770645, 'canton': 'VD'},   # new
    'Pompales': {'id': 10, 'lat': 46.666568, 'lng': 6.503529, 'canton': 'VD'},  # new
    'L\'Abbaye': {'id': 11, 'lat': 46.649326, 'lng': 6.320007, 'canton': 'VD'},  # new
    'Sainte-Croix': {'id': 12, 'lat': 46.821755, 'lng': 6.502249, 'canton': 'VD'},  # new
    'Château d\'Oex': {'id': 13, 'lat': 46.478255, 'lng': 7.141267, 'canton': 'VD'},

    # NE
    'Neuchâtel': {'id': 14, 'lat': 46.996007, 'lng': 6.944550, 'canton': 'NE'},
    'La Chaux-de-Fonds': {'id': 15, 'lat': 47.087581, 'lng': 6.809231, 'canton': 'NE'},
    'Malviliers': {'id': 16, 'lat': 47.031679, 'lng': 6.868278, 'canton': 'NE'},
    'Val-de-Travers': {'id': 17, 'lat': 46.924731, 'lng': 6.632067, 'canton': 'NE'}

}


class Environment(gym.Env):
    """
    Environment

    Attributes
    __________
    ambu_in_stations : dict
        Ambulances in each station.
    busy_ambulances : list
        List of busy ambulances, contains tuples with the station of prominence of the ambulance and the time when it
        will be available again.
    last_obs :
        observation of the previous state
    new_obs :
        observation of the new state
    incidents_generator : IncidentsGenerator()
        class with the method new_emergency(), returning the necessary information of the next emergency

    Methods
    _______
    reward(action)
        returns the reward of an action
    add_time(ini_time)
        time when the ambulance will be available again
    step(action)
        execute one time step within the environment
    reset()
        reset the state of the environment to an initial state
    next_observation()
        gets the next emergency and updates the busy ambulances
    render()
        render the environment
    """

    def __init__(self):
        super(Environment, self).__init__()

        self.ambu_in_stations = pickle.load(open('Data/Data_preprocessed/ambu_in_stations.pkl', 'rb'))
        self.busy_ambulances = []
        self.last_obs = None
        self.new_obs = None
        self.incidents_generator = IncidentsGenerator()

    def reward(self, action):
        """
        Calculate the reward of an action

        Parameters
        __________
        action : int from 0 to 17
            action

        Returns
        _______
        float
            reward of the action
        """

        # Weight, will be given by the priority (P0 -> 100, P1 -> 10, P2 -> 1)
        priority = self.last_obs[1]
        weight = 1 + 99 * (priority == 0) + 9 * (priority == 1)

        distances = self.last_obs[3]
        val = min(distances)/distances[action]

        reward = weight * (2 * (val ** 3) - 1)

        return reward

    def add_time(self, ini_time):
        """
        Adds the time the ambulance is busy to get the time that the ambulance will be free again

        Parameters
        __________
        ini_time : timedelta.timedelta()
            initial time

        Returns
        _______
        timedelta.timedelta()
            final time
        """

        duration = np.random.gamma(3.9110986587617522, 1109.3708330737102)  # Sample duration following gamma distribution

        end_time = ini_time + timedelta(seconds=duration)  # Add duration

        return end_time

    def step(self, action):
        """
        Execute one time step within the environment. This method is the one called to obtain experiences in the
        training.

        Parameters
        __________
        action : int (from 0 to 17 or 18)
            action done in the time step, from 0 to 17 correspond in sending ambulances from stations, 18 corresponds to
            do not do anything with very negative reward.

        Returns
        _______
        4-tuple
            next observation
        float
            reward
        boolean
            if the environment is finished, for this problem it will be always False
        dict
            current position of the ambulances
        {}
            extra information
        """

        self.last_obs = self.new_obs

        if action < 18:  # action 18 -> do nothing
            self.ambu_in_stations[action]['num'] -= 1
            ini_time = self.last_obs[0]
            end_time = self.add_time(ini_time)
            self.busy_ambulances.append((action, end_time))

            reward = self.reward(action)

        else:
            reward = -10000000000

        done = False

        self.new_obs = self.next_observation()

        return self.new_obs, reward, done, self.ambu_in_stations, {}

    def reset(self):
        """
        Reset the state of the environment to an initial state.

        Returns
        _______
        4-tuple
            next observation
        dict
            current position of the ambulances
        """

        self.ambu_in_stations = pickle.load(open('Data/Data_preprocessed/ambu_in_stations.pkl', 'rb'))
        self.busy_ambulances = []
        self.incidents_generator.reset()
        self.new_obs = self.next_observation()

        return self.new_obs, self.ambu_in_stations

    def next_observation(self):
        """
        Gets the new emergency and updates the busy ambulances and the number of available ambulances
        Returns
        _______
        4-tuple
            new emergency returned by the incident generator consisting of time, priority, coordinates, distances
        """

        emergency = self.incidents_generator.new_emergency()

        # update self.busy_ambulances
        indexes = [i for i in range(len(self.busy_ambulances))]
        for i in sorted(indexes, reverse=True):
            if self.busy_ambulances[i][1] < emergency[0]:
                self.ambu_in_stations[self.busy_ambulances[i][0]]['num'] += 1
                self.busy_ambulances.pop(i)

        return emergency

    def render(self):
        """
        Render the environment, consisting on the observation, number of ambulances per station and list of busy
        ambulances to a returned string, which can be printed in the screen or in a separate file.
        """

        string = ''
        string += '----------------------------------------\n'
        string += 'OBS:' + str(self.new_obs) + '\n'
        string += 'Station status:\n'
        aux = []
        for k, v in self.ambu_in_stations.items():
            aux.append('stn:' + str(k) + ', num:' + str(v['num']))
        string += str(aux) + '\n'
        string += 'Busy ambulances:' + str(self.busy_ambulances) + '\n'
        string += '----------------------------------------\n'

        return string

    

