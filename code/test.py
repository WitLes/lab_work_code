import numpy as np

probobility_matrix = np.zeros((10,10), dtype=float)
probobility_matrix[3][4] = 8
probobility_matrix *=(1/6)
print(probobility_matrix.max())
