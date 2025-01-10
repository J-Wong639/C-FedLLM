import numpy as np
import torch 

W=torch.randn(4096,32)

# 假设我们有一个权重矩阵的奇异值数组

singular_values=np.array([1,0.341911442,0.213011306,0.16537403,0.132666829,0.129365371,0.113703125,0.095099938,0.085732307,0.073890965,0.071437393,0.068748,0.054363823,0.053408645,0.048782916,0.042741884,0.03952176,0.036241585,0.034538845,0.032956778,0.030146681,0.028159609,0.026689107,0.022265036,0.017968524,0.016350742,0.014664556,0.013243866,0.011463283,0.010241507,0.008030483,0.005258585])

# 计算全秩的数量
full_rank_count = len(singular_values)
singular_values=singular_values/singular_values.max()

print(singular_values)

# 目标有效秩减少比率
target_reduction_ratio = 0.7

# 生成搜索空间
search_space = np.arange(0, 1, 0.005)

# 初始化当前秩减少比率
current_reduction_ratio = 0

# 初始化阈值
threshold = 0
# print(search_space)

# 搜索合适的阈值
for threshold in search_space:
    # 计算小于当前阈值的奇异值数量
    reduced_rank = np.sum(singular_values < threshold)
    
    # 计算当前秩减少比率
    current_reduction_ratio = reduced_rank / full_rank_count
    print("current_ratio=",current_reduction_ratio)
    
    # 检查当前秩减少比率是否在目标秩减少比率的容忍范围内,0.03125=1/32
    if (target_reduction_ratio + 0.03125) >= current_reduction_ratio >= (target_reduction_ratio - 0.03125):
        print(f"Found threshold: {threshold}")
        break





# 应该是找要压缩的比例，给定一个threshold，然后去搜索要压缩多少rank



target_threshold=0.175



filter_singular_values=singular_values[singular_values>target_threshold]

print(filter_singular_values)