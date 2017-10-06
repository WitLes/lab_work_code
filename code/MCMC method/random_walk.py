# coding=utf-8

import numpy as np
import math
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

size = 10000
rand_a = np.random.rand(size) - np.ones(size) / 2
axis_x = np.array([i for i in range(size)])

summ = np.zeros(size)
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(rand_a, bins=100)

for i in range(1, len(rand_a)):
    summ[i] = summ[i - 1] + rand_a[i]

plt.subplot(122)
plt.plot(axis_x, summ)
plt.show()
