"""
Code used to obtain the graphical evolution of the rewards, evolution of rewards per priority, 
and evolution of number of unsafe actions. 
"""

from main import *
from matplotlib import pyplot as plt


# Get episode of length 10000
'''
env = Environment()
env.reset()
num_incidents = env.incidents_generator.num_incidents
np.random.seed(1)
seeds = np.random.randint(0, num_incidents-10000, 1)
for alg in algorithms:
    print(alg)
    algorithms[alg]['test'](env, seeds, ep_length=10000)
'''
seeds = [128037]

# All rewards and priotities
all_rewards = []
all_p0 = []
all_p1 = []
all_p2 = []
for alg in ['Random', 'Greedy', 'LinearQL', 'DQN', 'DDQN', 'sac', 'smbpo']:
    rewards = []
    p0 = 0
    p1 = 0
    p2 = 0
    ap0 = []
    ap1 = []
    ap2 = []
    file = 'Results/' + alg + '_' + str(seeds[0]) + '.txt'
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == 'O':
            split = line.split('), ')
            priority = int(split[1].split(',')[0])
        if line[0] == 'A':
            split = line.split()
            rewards.append(float(split[6]))
            if priority == 0:
                p0 += float(split[3])
                ap0.append(p0)
            elif priority == 1:
                p1 += float(split[3])
                ap1.append(p1)
            else:
                p2 += float(split[3])
                ap2.append(p2)
    print(alg)
    all_rewards.append(rewards)
    print('Total', rewards[-1])
    all_p0.append(ap0)
    print('P0', rewards[-1])
    all_p1.append(ap1)
    print('P1', rewards[-1])
    all_p2.append(ap2)
    print('P2', rewards[-1])

# Plot evolution rewards
for rewards in all_rewards:
    plt.plot(rewards)
plt.legend(['Random', 'Greedy', 'LinearQL', 'DQN', 'DoubleDQN', 'SAC', 'SMBPO'])
plt.show()
plt.close()

# Plot evolution P0
for rewards in all_p0:
    plt.plot(rewards)
plt.legend(['Random', 'Greedy', 'LinearQL', 'DQN', 'DoubleDQN', 'SAC', 'SMBPO'])
plt.show()
plt.close()

# Plot evolution P1
for rewards in all_p1:
    plt.plot(rewards)
plt.legend(['Random', 'Greedy', 'LinearQL', 'DQN', 'DoubleDQN', 'SAC', 'SMBPO'])
plt.show()
plt.close()

# Plot evolution P2
for rewards in all_p2:
    plt.plot(rewards)
plt.legend(['Random', 'Greedy', 'LinearQL', 'DQN', 'DoubleDQN', 'SAC', 'SMBPO'])
plt.show()
plt.close()


# Unsafe actions
all_counts = []
for alg in ['Random', 'Greedy', 'LinearQL', 'DQN', 'DDQN', 'sac', 'smbpo']:
    count = 0
    acounts = []
    file = 'Results/' + alg + '_' + str(seeds[0]) + '.txt'
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == '[':
            amb_status = line.split()
        if line[0] == 'A':
            action = int(line.split()[1])
            if amb_status[2*(action+1)-1][-4:-2] == ':0':
                count += 1
            acounts.append(count)
    all_counts.append(acounts)
    print(alg, count)
# Plot evolution unsafe actions
for counts in all_counts:
    plt.plot(counts)
plt.legend(['Random', 'Greedy', 'LinearQL', 'DQN', 'DoubleDQN', 'SAC', 'SMBPO'])
plt.show()
plt.close()

