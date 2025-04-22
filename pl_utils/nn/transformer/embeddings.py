import torch
from torch import nn
from torch import Tensor


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


class RotaryPosEmbedding(nn.Module):
    """
    改编自 transformers 库中 gemma 模型的 modeling_gemma.py
    """

    def __init__(self, dim, base=10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        q,
        k,
        position_ids=None,
    ):
        """
        q 和 k 输入维度：[batch_size, num_heads, seq_len, dim]
        """

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        if position_ids is None:
            b, h, n, d = q.shape
            position_ids = torch.arange(n, device=q.device).expand(b, -1)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        device_type = q.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class TimeStepSinusoidalEmbbedding(nn.Module):
    """
    将时间步转换为正余弦嵌入
    """

    def __init__(
        self,
        dim,
        base=10000,
    ):
        super().__init__()
        assert dim % 2 == 0, 'dim must be even'
        self.dim = dim
        self.register_buffer('base', torch.tensor(base))

    def forward(self, timesteps: Tensor):
        """

        Args:
            timesteps: Tensor: 整数张量，形状为 [batch, seq_len]
        """
        half_dim = self.dim // 2
        exponent = (
            -torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            / (half_dim - 1)
            * torch.log(self.base)
        )
        frequencies = torch.exp(exponent)
        angles = timesteps.unsqueeze(-1).float() * frequencies
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # 最终形状为 [batch, seq_len, dim]
        return emb
