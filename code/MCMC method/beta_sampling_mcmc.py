import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt


def beta_s(x, a, b):
    return x ** (a - 1) * (1 - x) ** (b - 1)


def beta(x, a, b):
    return beta_s(x, a, b) / ss.beta(a, b)


def plot_mcmc(a, b):
    cur = np.random.rand()
    states = [cur]
    for i in range(50000):
        u = np.random.rand()
        next = np.random.rand()
        if u < np.min((beta_s(next, a, b) / beta_s(cur, a, b), 1)):
            states.append(next)
            cur = next
    print(len(states))   # 每一次实验最终接受的数据量是不一定的，states的长度也是不一定的
    x = np.arange(0, 1, 0.01)
    plt.plot(x, beta(x, a, b), lw=2)
    plt.hist(states[-4000:], bins=50, normed=True)  # 截取后4000个数据，保证数据量的一致性

if __name__ == '__main__':
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plot_mcmc(0.1, 0.1)
    plt.subplot(132)
    plot_mcmc(1, 1)
    plt.subplot(133)
    plot_mcmc(2, 3)
    plt.show()