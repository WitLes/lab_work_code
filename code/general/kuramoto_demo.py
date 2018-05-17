import numpy as np
import matplotlib.pyplot as plt

N = 20
iter_num = 100
theta = [2*np.pi*np.random.random() for i in range(N)]
data = [[0 for i in range(N)] for i in range(iter_num)]

for i in range(iter_num):

	for j in range(N):
		zgma = 0

		for k in range(N):
			zgma += 0.1*np.sin(theta[k] - theta[j])

		data[i][j] = theta[j] + zgma

	theta = data[i]

iter_data = np.transpose(data)

for i in range(N):
	plt.plot(iter_data[i])
	print(data[i][1])
plt.show()


