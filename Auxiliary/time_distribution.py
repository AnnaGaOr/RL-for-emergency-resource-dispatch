"""
File used to calculate the parameters of the gamma distribution for the busy times.
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def duration (initial, final):
    i_year, i_month, i_day, i_hour, i_minute, i_second = read_date(initial)
    f_year, f_month, f_day, f_hour, f_minute, f_second = read_date(final)

    i_date = datetime(i_year, i_month, i_day, i_hour, i_minute, i_second)
    f_date = datetime(f_year, f_month, f_day, f_hour, f_minute, f_second)

    return (f_date-i_date).seconds


def read_date(date):
    aux1 = date.split('/')
    month = int(aux1[0])
    day = int(aux1[1])
    aux2 = aux1[2].split(' ')
    year = int(aux2[0])
    aux3 = aux2[1].split(':')
    hour = int(aux3[0])
    minute = int(aux3[1])
    second = int(aux3[2])
    ampm = aux2[2]
    if (hour == 12 and ampm == 'AM') or (hour != 12 and ampm == 'PM'):
        hour += 12
        hour = hour % 24

    return year, month, day, hour, minute, second

directory = 'Data/Resources'
times = []
for file in os.listdir(directory):
    resources = pd.read_csv(os.path.join(directory, file), dtype=str)
    for i in range(len(resources)):
        if resources['Text'][i] == 'Demande d\'engagement':
            initial = resources['LocalTime'][i]
        elif resources['Text'][i] == 'Terminé (en rue)':
            final = resources['LocalTime'][i]
            times.append(duration(initial, final))


times = np.array(times)
mean = np.mean(times)
std = np.std(times)

print('Mean =   ', mean)  # Mean =    4338.858777303995
print('Std =    ', std)  # Std =     2193.946985769463

sigma = std ** 2 / mean
k = mean / sigma

print('k =  ', k)  # k =   3.9110986587617522
print('Sigma =  ', sigma)  # Sigma =   1109.3708330737102

# Plot histogram
plt.hist(times, bins=100, density=True)

# Plot gamma distribution
x = np.linspace(0, max(times), 10000)
y = stats.gamma.pdf(x, a=k, scale=sigma)
plt.plot(x, y)

plt.xlim([0, 40000])
plt.show()
plt.close()
