import torch

# 假设 A 是一个已定义的张量
A = torch.tensor([[3, 3], [3, 3],[3,3]])

# 生成与 A 维度相同的单位矩阵
B = A/3

print(B)
