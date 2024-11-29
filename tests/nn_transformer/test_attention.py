import unittest
from pl_utils.nn import GQASelfAttention
import torch


class TestTransformerAttention(unittest.TestCase):
    def test_gqa_self_attention_forward(self):
        attn = GQASelfAttention(
            hidden_size=16,
            mlp_hidden_size_factor=2,
            head_dim=8,
            num_heads=2,
            num_kv_heads=1,
            attn_dropout=0.1,
            attn_bias=False,
            attn_implementation="scaled_dot_product_attention",
        )
        x = torch.randn(8, 10, 16)
        attn_mask = torch.randn(8, 2, 10, 10)
        y = attn(x, attn_mask=attn_mask)
        self.assertEqual(y.shape, (8, 10, 16))


if __name__ == '__main__':
    unittest.main()
