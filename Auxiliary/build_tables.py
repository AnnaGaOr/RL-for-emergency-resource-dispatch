"""
This has been used to build the tables with the episodes for the appendix of the report.
To generate more tables indicate the path of the file in the variable file.
"""

file = '../Results/smbpo_304137.txt'

with open(file) as f:
    lines = f.readlines()

print('\hline \\rowcolor{gray} Data and Time & Priority & \multicolumn{18}{l|}{Distances / Available Ambulances}  & Action & Reward & Accumulated \\\\ ')
print('\\rowcolor{gray} &  & 1&2&3&4&5&6&7&8&9&10&11&12&13&14&15&16&17&18 &  &  & Reward\\\\ \hline')

for i in range(int(len(lines)/7)):

    obs = lines[7*i + 1]
    obs_split = obs.split()
    while(len(obs_split) < 27):
        obs_split.insert(6-27, '0),')
        obs_split[4-27] = obs_split[4-27][:-2] + ','

    # Date and time
    year = int(obs_split[0][-5:-1])
    month = int(obs_split[1][:-1])
    day = int(obs_split[2][:-1])
    hour = int(obs_split[3][:-1])
    min = int(obs_split[4][:-1])
    sec = int(obs_split[5][:-2])
    local_time = str(year * 10 ** 10 + month * 10 ** 8 + day * 10 ** 6 + hour * 10 ** 4 + min * 10 ** 2 + sec)
    date = local_time[6:8] + '/' + local_time[4:6] + '/' + local_time[0:4]
    time = local_time[8:10] + ':' + local_time[10:12] + ':' + local_time[12:14]

    # Priority
    priority = obs_split[6][:-1]
    if priority != '0' and priority != '1':
        priority = '2'

    # Distances
    distances = obs_split[9][1:-1] + ' & '
    for j in range(16):
        distances += obs_split[10+j][:-1] + ' & '
    distances += obs_split[26][:-2]

    # Available ambulances
    stations = lines[7*i + 3]
    stations_split = stations.split(':')
    available = ''
    for j in range(17):
         available += stations_split[2*j + 2][:-7] + ' & '
    available += stations_split[36][:-3]

    # Action Reward Accumulated
    step = lines[7*i + 6]
    step_split = step.split()
    action = str(int(step_split[1]) + 1)
    reward = step_split[3]
    accumulated = step_split[6]

    if i % 2 == 0:
        print(date + ' & ' + priority + ' & ' + distances + ' & ' + action + ' & ' + reward + ' & ' + accumulated + ' \\\\')
        print(time + ' & & ' + available + ' & & & \\\\')
    else:
        print('\\rowcolor{lightgray} ' + date + ' & ' + priority + ' & ' + distances + ' & ' + action + ' & ' + reward + ' & ' + accumulated + ' \\\\')
        print('\\rowcolor{lightgray} ' + time + ' & & ' + available + ' & & & \\\\')

