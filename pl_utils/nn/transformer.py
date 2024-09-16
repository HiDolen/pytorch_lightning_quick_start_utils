import torch
from torch import nn


class LearnablePosEmbedding(nn.Module):
    """
    可学习位置编码。
    """
    def __init__(self, dim, max_seq_len):
        """
        
        Args:
            dim: 位置编码的维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.embeddings = nn.Embedding(max_seq_len, dim)
        self.register_buffer('position_ids', torch.arange(max_seq_len))

    def forward(self, x):
        position_ids = self.position_ids[: x.size(-2)]
        return x + self.embeddings(position_ids)
