import torch
import torch.nn as nn
from .linear import CustomLinear
from .silu import SiLU


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        建议：d_ff = 8 / 3 * d_model
        且d_ff要为64的整数倍
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = CustomLinear(d_model, d_ff)
        self.linear_gate = CustomLinear(d_model, d_ff)
        self.linear_2 = CustomLinear(d_ff, d_model)
        self.silu = SiLU()

    def forward(self, x):
        x_temp = self.linear_1(x)
        x_swish = self.silu(x_temp)
        x_gate = self.linear_gate(x)
        return self.linear_2(x_swish * x_gate)
