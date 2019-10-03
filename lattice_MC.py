import numpy as np
import matplotlib.pyplot as plt

p = 1/4.0
size = 18
lattice = np.zeros((size, size))
for i in range(size):
    for j in range(size):

        lattice[i, j] = (np.random.random() < p)

print(lattice)

def find_neighbor(center, cluster, bank):
    """ center: list containing [i, j] of the center particle
    cluster: a list of pairs (tuples) to store the current cluster
    bank: a list of pairs to store all clusters points already found"""
    i = center[0]
    j = center[1]
    flag = 0
    count = 1
    for k, l in [(1,-1), (1,0),(0,1),(-1,1),(-1,0),(0,-1)]:
        if i+k<size and i+k >= 0 and j+l< size and j+l>=0:
            if lattice[i+k, j+l] == 1 and (i+k, j+l) not in cluster:
                flag = 1
                count += 1
                cluster.append((i + k, j + l))
                bank.append((i + k, j + l))
                find_neighbor([i+k, j+l], cluster, bank)
    if flag == 0:
        return count


bank = []
data = []
for i in range(size):
    for j in range(size):
        if lattice[i, j] == 1 and (i, j) not in bank:
            cluster = []
            find_neighbor([i, j], cluster, bank)
            data.append(len(cluster))
            print(cluster)

print(data)
hist, bin_edges = np.histogram(data)
plt.figure()
plt.hist(data)
plt.show()