import numpy as np

import matplotlib.pyplot as plt

def draw_original_data_distribution():
    NUM = 10000
    np.random.seed(6)
    data = np.random.normal(2, 2, NUM)
    print(data)
    list_count = [0 for i in range(1000)]
    for value in data:
        list_count[int(value*50)+300] += 1
    plt.scatter([x for x in range(1000)],list_count)
    plt.show()

def add(a,b):
    sum = 0
    for i,j in a,b:
        sum += a*b
    return sum

print("1")