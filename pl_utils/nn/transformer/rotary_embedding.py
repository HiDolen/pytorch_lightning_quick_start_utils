from typing import Tuple, Union, List
import torch
from torch import nn


def get_latent_2d_ids(height, width):
    """用于生成 2D 位置编码。改编自 diffusers 库的 _prepare_latent_image_ids 函数。

    生成的编码假设将二维序列 squeeze 成一维。

    Args:
        height (`int`):
        width (`int`):

    Returns:
        `torch.Tensor`: [height * width, 2]
    """
    latent_img_ids = torch.zeros(height, width, 2)
    latent_img_ids[..., 0] = latent_img_ids[..., 0] + torch.arange(height)[:, None]
    latent_img_ids[..., 1] = latent_img_ids[..., 1] + torch.arange(width)[None, :]

    ids_height, ids_width, ids_channels = latent_img_ids.shape

    latent_img_ids = latent_img_ids.reshape(ids_height * ids_width, ids_channels)

    return latent_img_ids


class RotaryEmbeddingMultiDimension(nn.Module):
    """多维度旋转位置编码。

    能够返回多维度 RoPE 的 cos 和 sin，可用于生成像是 ViT 的二维位置编码。

    生成的 RoPE 编码需要在 attention 环节借助 `apply_rotary_emb()` 向 kv 施加。
    """

    def __init__(self, theta: int = 10000, dim: List[int] = [64, 64]):
        """初始化多维旋转位置编码。

        Args:
            theta (`int`): 设置 theta 值
            dim (`List[int]`): 每个维度分别的 channel 数。sum(dim) 应等于单个注意力头的维度数。\
                会间接影响 RoPE 所表示维度的数量。
        """
        super().__init__()
        self.theta = theta
        self.dim = dim

    def get_1d_rotary_pos_embed(self, pos_ids, dim):
        """改编自 diffusers 库的 get_1d_rotary_pos_embed 函数。"""
        freqs = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        freqs = 1.0 / (self.theta**freqs)  # [D/2]
        freqs = torch.outer(pos_ids, freqs)  # type: ignore   # [S, D/2]

        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        dim_count = pos_ids.size(-1)
        cos_out, sin_out = [], []
        pos_ids = pos_ids.float()
        for i in range(dim_count):
            cos, sin = self.get_1d_rotary_pos_embed(pos_ids[:, i], self.dim[i])
            cos_out.append(cos)
            sin_out.append(sin)
        cos = torch.cat(cos_out, dim=-1).to(device=pos_ids.device)
        sin = torch.cat(sin_out, dim=-1).to(device=pos_ids.device)
        return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """施加旋转位置编码。改编自 diffusers 库的 apply_rotary_emb 函数。

    Args:
        x (`torch.Tensor`): [B, H, S, D]
        freqs_cis (`Tuple[torch.Tensor]`): 预先准备好的旋转位置编码。([S, D], [S, D],)
    """
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    # rotate half
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out
