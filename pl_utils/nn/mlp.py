import torch
import torch.nn as nn

class GeGLUMLP(nn.Module):
    """
    GeGLU MLP，使用 GELU 激活函数的 GLU 完成的的 MLP。代码来自于 Google 的 Gemma 模型（https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/）。

    根据论文 GLU Variants Improve Transformer 的实验，GeGLU 和 SwiGLU 能带来优于 ReLU、GLU、GELU 等的性能。
    """
    def __init__(
        self,
        hidden_size,
        intermediate_dim,
    ):
        """
        
        Args:
            hidden_size: 输入到该层的维度
            intermediate_dim: 中间维度。习惯上设置为 hidden_size 的 4 倍
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_size, bias=False)

        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))