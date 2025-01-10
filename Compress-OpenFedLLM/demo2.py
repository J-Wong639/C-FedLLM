import numpy as np
import torch 







matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
U, S, V = np.linalg.svd(matrix)
print(S)

