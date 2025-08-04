import torch
import torch.nn as nn


class CustomSoftmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 指定dim的时候，torch.max会返回最大值和索引，这里仅用到最大值
        # x_max, x_max_indices = torch.max(x, dim=self.dim, keepdim=True)
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        # 用exp(x-x_max)代替exp(x)，避免溢出，保证数值稳定性
        exp_x = torch.exp(x - x_max)
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp_x
