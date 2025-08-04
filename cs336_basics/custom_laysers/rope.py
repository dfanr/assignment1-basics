import torch
import torch.nn as nn


class CustomRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.register_buffer('r_complex', self.init_r().to(self.device))

    def init_r(self):
        theta_i = torch.arange(0, self.max_seq_len, device=self.device).float()  # shape: (max_seq_len, )
        theta_k = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device).float() / self.d_k))  # shape: (d_k, )
        theta_k_i = torch.outer(theta_i, theta_k).float()  # shape: (max_seq_len, d_k, )
        # torch.polar构造复数tensor，第一个参数是模（R=1），第二个参数是角度（angle）
        # Z = r * e ^ (i * theta) = r * (cos(theta) + i*sin(theta))
        # 说人话：将cos值存到复数的实部，将sin值存到了复数的虚部
        theta_complex = torch.polar(torch.ones_like(theta_k_i), theta_k_i)
        return theta_complex

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        current_seq_len = x.shape[-2]
        current_d_k = x.shape[-1]
        assert current_d_k % 2 == 0
        pre_dims = x.shape[:-1]
        assert current_d_k == self.d_k
        x_complex = torch.view_as_complex(x.view(*pre_dims, current_d_k // 2, 2))  # shape: (batch, n_heads, seq_len, head_dim//2)
        # 旋转：复数乘法
        x_rotated = self.r_complex[:current_seq_len, :] * x_complex[..., token_positions, :]
        x_rotated_real = torch.view_as_real(x_rotated)  # (batch, n_heads, seq_len, head_dim//2, 2)
        return x_rotated_real.view(*pre_dims, current_d_k)

