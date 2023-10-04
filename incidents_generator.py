"""
This file is to generate new incidents. At every step, the class environment uses the method new_emergency() which
returns the time, the priority, the coordinates and the distances to each station of the next emergency of the dataset.
"""

import numpy as np
import pickle
from datetime import datetime


class IncidentsGenerator:
    """
    This class is used to get the new emergency for the next observation.

    Attributes
    __________
    incidents : numpy.array()
        dataset
    num_incidents : int
        number of incidents ( = incidents.shape[0])
    idx : int (from 0 to num_incidents)
        index of the current accident

    Methods
    _______
    new_emergency()
        get next experience
    reset()
        new random index
    """

    def __init__(self):

        self.incidents = pickle.load(open('Data/Data_preprocessed/all_prep_inci_with_times.pkl', 'rb'))
        self.num_incidents = self.incidents.shape[0]
        self.idx = np.random.randint(0, self.num_incidents)
        
    def new_emergency(self):
        """
        Returns the next experience with its time, priority, coordinates and distances to the stations

        Returns
        _______
        datetime.datetime()
            date and time of the new emergency
        int
            priority level of the new emergency
        tuple of floats
            Cartesian coordinates of the new emergency location
        list of floats
            distances to each station
        """

        found = False
        while not found:

            # time
            time = self.incidents[self.idx, 0]
            aux = str(time)
            time = datetime(int(aux[0:4]), int(aux[4:6]), int(aux[6:8]), int(aux[8:10]), int(aux[10:12]), int(aux[12:14]), 0)

            # priority
            priority = int(self.incidents[self.idx, 1])

            # coordinates
            coordinates = self.incidents[self.idx, 2]

            # distances
            distances = self.incidents[self.idx, 3]

            # next index
            self.idx += 1
            if self.idx == self.num_incidents:
                self.reset()

            if not distances[0] is None:  # in case there was some error finding the routes we eliminate it
                found = True

        return time, priority, coordinates, distances

    def reset(self):
        """
        New random index.
        """

        self.idx = np.random.randint(0, self.num_incidents)

    