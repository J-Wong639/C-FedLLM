import numpy as np

# 创建一个方阵
A = np.array([[1, 2], [3, 4]])

# 计算矩阵A的逆
A_inv = np.linalg.inv(A)

print("矩阵A的逆：")
print(A_inv)

A_pinv=np.linalg.pinv(A)
print("矩阵A的伪逆：")
print(A_pinv)
