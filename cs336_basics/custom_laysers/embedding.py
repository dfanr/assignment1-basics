import torch
import torch.nn as nn


class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = self._init_weight()

    def _init_weight(self):
        weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        # 初始化
        std = 1.0
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return weight

    def forward(self, x: torch.LongTensor):
        return self.weight[x]
