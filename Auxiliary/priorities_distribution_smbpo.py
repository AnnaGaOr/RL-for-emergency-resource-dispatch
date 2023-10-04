"""
This file is used to estimate the probability of each priority for a new accident in dynamics from SMBPO.

Each probability is calculated as the proportion of that priority over all the emergencies
"""

import numpy as np
from incidents_generator import *

ig = IncidentsGenerator()
ig.idx = 0

priorities = np.zeros(ig.num_incidents)

for i in range(ig.num_incidents):
    priorities[i] = int(ig.new_emergency()[1])  # Priority of each incident

# Calculate the proportion of emergencies with priority 0, 1, 2 or bigger

p0 = np.sum(priorities == 0)
p1 = np.sum(priorities == 1)
p2 = ig.num_incidents - p0 - p1

print('P=0:', p0, p0/ig.num_incidents)
print('P=1:', p1, p1/ig.num_incidents)
print('P=2:', p2, p2/ig.num_incidents)

'''
P=0: 89630 0.2718828868180936
P=1: 139874 0.4242926130848379
P=2: 100160 0.3038245000970685
'''