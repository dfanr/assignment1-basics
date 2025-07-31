import torch
import torch.nn as nn


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 创建参数
        self.weight = self._init_weight()

        # 不使用bias
        # self.bias = self._init_bias()

    def _init_weight(self):
        weight = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        # 初始化
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return weight

    def _init_bias(self):
        bias = nn.Parameter(torch.zeros(self.out_features, device=self.device, dtype=self.dtype))
        return bias

    def forward(self, x):
        return x @ self.weight.T
