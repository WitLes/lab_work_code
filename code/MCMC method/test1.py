# coding=utf-8

import numpy as np
import math
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

rand_a = np.random.rand(100000)
rand_b = np.random.rand(100000)
print(rand_a)
print(rand_b)

result1 = np.sqrt(-2 * np.log(rand_a)) * np.cos(2 * math.pi * rand_b)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(result1, bins=100)
plt.show()
