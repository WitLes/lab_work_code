# -*- coding=utf8 -*-

# Code from Chapter 14 of Machine Learning: An Algorithmic Perspective
# The basic rejection sampling algorithm
import numpy as np
import matplotlib.pyplot as plt


def qsample():
    return np.random.rand() * 4.


def p(x):
    return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)


def rejection(nsamples):
    M = 0.72  # 0.8
    samples = np.zeros(nsamples, dtype=float)
    count = 0
    for i in range(nsamples):
        accept = False
        while not accept:
            x = qsample()
            u = np.random.rand() * M
            if u < p(x):
                accept = True
                samples[i] = x
            else:
                count += 1
    print(count)
    return samples


x = np.arange(0, 4, 0.01)
realdata = 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)

plt.plot(x, realdata, color='black')

samples = rejection(10000)
plt.hist(samples, normed=1, bins=100, color='b')

plt.show()
