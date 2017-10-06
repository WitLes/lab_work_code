# coding=utf-8

import numpy as np
import math
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

sampleNo = 10000
# 一维正态分布
# 下面三种方式是等效的
rand_a = np.random.rand(10)
mu = 3
sigma = 1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo)
print(s)
plt.subplot(121)  # 1行 2列 第一个
plt.hist(s, 50, normed=True)

# 二维正态分布
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
s = np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.subplot(122)  # 1行 2列 第二个
# 注意绘制的是散点图，而不是直方图
plt.plot(s[:, 0], s[:, 1], '+')
plt.show()

# 利用均匀分布生成正态分布

rand_a = np.random.rand(100000)
rand_b = np.random.rand(100000)
print(rand_a)
print(rand_b)

result1 = np.sqrt(-2 * np.log(rand_a)) * np.cos(2 * math.pi * rand_b)
result2 = np.sqrt(-2 * np.log(rand_a)) * np.sin(2 * math.pi * rand_b)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(result1, bins=100)
plt.show()
