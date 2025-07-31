import torch
import torch.nn as nn


class CustomRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = self._init_g()

    def _init_g(self):
        g = nn.Parameter(torch.ones((self.d_model,), device=self.device, dtype=self.dtype))
        return g

    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        rms_x = torch.sqrt(torch.mean(x ** 2 + self.eps, dim=-1, keepdim=True))
        result = x / rms_x * self.g

        return result.to(in_type)
