import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import RMSNorm

from pl_utils.nn import GeGLUMLP


class GQASelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size=32,
        mlp_hidden_size_factor=4,
        head_dim=8,
        num_heads=4,
        num_key_value_heads=1,
        attn_dropout=0.,
        attn_bias=False,
        attn_implementation="scaled_dot_product_attention",
    ):
        """
        一个进行自注意力的多头注意力实现。
        
        Args:
            hidden_size: 隐藏层的维度
            mlp_hidden_size_factor: MLP 维度相对隐藏层维度的倍数
            head_dim: 头维度
            num_heads: 头数量
            num_key_value_heads: kv 头数量
            attn_dropout: dropout
            attn_bias: 线性层是否启用 bias
            attn_implementation: 注意力的实现
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads

        self.attn_dropout = attn_dropout
        self.attn_dropout_origin = attn_dropout
        self.attn_bias = attn_bias

        self.attention_implementation = attn_implementation

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attn_bias)

        self.mlp = GeGLUMLP(hidden_size, hidden_size * mlp_hidden_size_factor)

        self.input_layernorm = RMSNorm(hidden_size)
        self.post_layernorm = RMSNorm(hidden_size)

    def forward(self, hidden_states, attn_mask=None):
        """
        
        Args:
            hidden_states: [b_size, f_size, hidden_size]
            attn_mask: [b_size, f_size, f_size]
        """
        b_size, f_size, _ = hidden_states.size()
        # attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        q_states = q_states.view(b_size, f_size, self.num_heads, self.head_dim)
        k_states = k_states.view(b_size, f_size, self.num_key_value_heads, self.head_dim)
        v_states = v_states.view(b_size, f_size, self.num_key_value_heads, self.head_dim)
        q_states = q_states.transpose(1, 2)  # [b_size, num_heads, f_size, head_dim]
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        if self.attention_implementation == "scaled_dot_product_attention":
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
                attn_mask=attn_mask,
                enable_gqa=True,
            )
        else:
            attn_output = self.attention_implementation(
                q_states,
                k_states,
                v_states,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
                attn_mask=attn_mask,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b_size, f_size, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        hidden_states = residual + attn_output

        # fc
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
