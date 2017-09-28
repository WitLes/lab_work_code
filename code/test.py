import networkx as nx  
import matplotlib.pyplot as plt  
import random
import numpy as np

random_seed = np.random.gamma(2,3,1000)
print(random_seed)

plt.hist(random_seed, bins=30,normed=True)
plt.show()
