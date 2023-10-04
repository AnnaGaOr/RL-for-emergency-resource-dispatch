"""
Code used to finish the preprocessing.
"""
import numpy as np
import pickle

from time_models.OpenRouteTravelTimeModel import OpenRouteTravelTimeModel


data = pickle.load(open('../Data/Data_preprocessed/all_prep_inci.pkl', 'rb'))

num = len(data)
times = data[:, 1]
priorities = data[:, 2].astype(np.float).astype(int)
coordinates = data[:, 3]

dataset = np.zeros((num, 4), dtype=object)
dataset[:, 0] = times
dataset[:, 1] = priorities
dataset[:, 2] = coordinates

# Calculate times

tm = OpenRouteTravelTimeModel()

for i in range(num):
    if i % 100 == 0:
        print('Step', i, 'of', num)
    coords = (coordinates[i][1], coordinates[i][0])
    dataset[i, 3] = tm._travel_times(coords)[:-1]

# Save whole dataset
pickle.dump(dataset, open('../Data/Data_preprocessed/all_prep_inci_with_times.pkl', 'wb'))
