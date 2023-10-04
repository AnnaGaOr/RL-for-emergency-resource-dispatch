"""
This file is used to estimate the parameter lambda, of the poisson process for the occurrence of the next accident in
SMBPO

To estimate when the next emergency will occur I will assume that it follows a poisson process
# http://www.math.uchicago.edu/~may/VIGRE/VIGRE2010/REUPapers/Mcquighan.pdf
Lambda is defined as the mean number of incidents per second
"""

import numpy as np
from matplotlib import pyplot as plt

from incidents_generator import *

ig = IncidentsGenerator()

num_incidents = ig.num_incidents  # Total number of incidents

times = []
ig.idx = 0
t0 = ig.new_emergency()[0]
for i in range(329657):
    t = ig.new_emergency()[0]
    times.append((t-t0).total_seconds())
    t0 = t

plt.hist(times, bins=100, density=True)

l = 1/np.mean(times)
print('lambda =', l)

x = np.linspace(0, 5000, 10000)
y = l * np.exp(-l * x)
plt.plot(x, y)
plt.xlim([0, 5000])
plt.show()
plt.close()

'''
lambda = 0.0016781755212247
'''
