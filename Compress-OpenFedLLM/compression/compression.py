import torch
import numpy as np
import math






class QSGDCompressor(object):
    # 定义QSGD压缩器类
    def __init__(self):
        self.name = 'qsgd'  # 定义压缩器的名称为'qsgd'
        self.residuals = {}  # 初始化一个空字典，用于存储各个张量的残差
        self.values = {}  # 初始化一个空字典，用于存储各个张量的压缩后的值
        self.zc = None  # 初始化zc属性为None，可能用于后续的某些操作
        self.current_ratio = 1  # 设置当前压缩比率为1，表示初始状态下没有压缩
        self.shapes = {}  # 初始化一个空字典，用于存储各个张量的形状信息

    def get_qsgd(self, x, s, is_biased=False):
        # 定义QSGD (Quantized Stochastic Gradient Descent) 方法
        norm = x.norm(p=2)  # 计算张量x的2范数（L2范数）
        
        level_float = s * x.abs() / norm  # 计算量化值：将张量x的每个元素绝对值乘以s，然后除以范数
        
        previous_level = torch.floor(level_float)  # 向下取整，得到每个元素的量化下界
        
        # 添加随机量化：生成与x形状相同的随机张量，用于决定是否将量化值向上调整一个级别
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        
        new_level = previous_level + is_next_level  # 最终的量化级别：previous_level加上随机调整值

        scale = 1  # 初始化缩放因子
        
        if is_biased:
            d = x.nelement()  # 获取张量x中元素的总数
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)  # 计算QSGD的方差界，用于减小量化误差
        
        # 返回量化后的结果
        return scale * torch.sign(x) * norm * new_level / s

    def qsgd_quantize_numpy(self, x, s, is_biased=False):
        """在绝对值系数上对张量x进行d级量化"""
        norm = np.sqrt(np.sum(np.square(x)))  # 计算x的2范数（L2范数）
        
        level_float = s * np.abs(x) / norm  # 计算量化值：将x的每个元素绝对值乘以s，然后除以范数
        
        previous_level = np.floor(level_float)  # 向下取整，得到每个元素的量化下界
        
        # 添加随机量化：生成与x形状相同的随机数组，用于决定是否将量化值向上调整一个级别
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        
        new_level = previous_level + is_next_level  # 最终的量化级别：previous_level加上随机调整值

        scale = 1  # 初始化缩放因子为1
        
        if is_biased:
            d = len(x)  # 获取x的长度
            scale = 1.0 / (np.minimum(d / s ** 2, np.sqrt(d) / s) + 1.0)  # 计算QSGD的方差界，用于减小量化误差
        
        # 返回量化后的结果
        return scale * np.sign(x) * norm * new_level / s

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        # 定义压缩方法，接受张量、名称、量化级别和是否有偏置作为参数
        if quantize_level != 32:
            s = 2 ** quantize_level - 1  # 计算量化参数s，为2的量化级别次方减1
            values = self.get_qsgd(tensor, s, is_biased)  # 使用QSGD方法进行压缩
        else:
            values = tensor  # 如果量化级别为32，则不进行压缩
        return values  # 返回压缩后的值（或未压缩的原始张量）

    def decompress_new(self, tensor):
        # 这个方法用于解压缩操作
        return tensor  # 直接返回输入的tensor，不进行任何处理

    def update_shapes_dict(self, tensor, name):
        # 更新shapes字典，记录不同名称的张量的形状信息
        self.shapes[name] = tensor.shape  # 将传入的tensor的形状存储到self.shapes字典中，以name为键




