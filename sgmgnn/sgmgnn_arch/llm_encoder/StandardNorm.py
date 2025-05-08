import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        
        # 初始化Normalize类，继承自nn.Module
        super(Normalize, self).__init__()
        self.num_features = num_features  # 这里相当于 N，即特征数
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):

        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):

        dim2reduce = (0, 3)  # 需要沿批次(B)和时间步(T)维度计算均值和标准差
        if self.subtract_last:
            self.last = x[:, :, :, -1].unsqueeze(-1)  # 取最后一个时间步，形状为 (B, N, 1, 1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # 计算均值，形状为 (1, N, 1, 1)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()  # 计算标准差

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last  # 减去最后一个时间步的值
        else:
            x = x - self.mean  # 减去均值
        x = x / self.stdev  # 除以标准差
        if self.affine:
            x = x * self.affine_weight.view(1, -1, 1, 1)  # 仿射变换
            x = x + self.affine_bias.view(1, -1, 1, 1)    # 添加偏置
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias.view(1, -1, 1, 1)  # 反仿射变换
            x = x / (self.affine_weight.view(1, -1, 1, 1) + self.eps * self.eps)
        x = x * self.stdev  # 恢复标准差
        if self.subtract_last:
            x = x + self.last  # 加回最后一个时间步的值
        else:
            x = x + self.mean  # 加回均值
        return x
