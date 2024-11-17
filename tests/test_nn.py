import unittest
from pl_utils.nn import (
    # MLP
    GeGLUMLP,
    # Transformer Embeddings
    LearnablePosEmbedding,
    RotaryPosEmbedding,
    TimeStepSinusoidalEmbbedding,
    # Transformer Attention
    GQASelfAttention,
)
import torch


class TestMLP(unittest.TestCase):
    def test_geglu_mlp_forward(self):
        mlp = GeGLUMLP(10, 40)
        x = torch.randn(32, 10)
        y = mlp(x)
        self.assertEqual(y.shape, (32, 10))


class TestTransformerEmbeddings(unittest.TestCase):
    def test_learnable_pos_embedding_forward(self):
        pos_emb = LearnablePosEmbedding(10, max_seq_len=100)
        x = torch.randn(32, 100, 10)
        y = pos_emb(x)
        self.assertEqual(y.shape, (32, 100, 10))

    def test_rotary_pos_embedding_forward(self):
        pos_emb = RotaryPosEmbedding(10, base=10000)
        q = torch.randn(32, 4, 100, 10)
        k = torch.randn(32, 4, 100, 10)
        q, k = pos_emb(q, k)
        self.assertEqual(q.shape, (32, 4, 100, 10))
        self.assertEqual(k.shape, (32, 4, 100, 10))

    def test_time_step_sinusoidal_embedding_forward(self):
        time_emb = TimeStepSinusoidalEmbbedding(6, base=100)
        times = torch.arange(4)
        embeddings = time_emb(times)
        embedding_target = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000],
                    [0.8415, 0.0998, 0.0100, 0.5403, 0.9950, 0.9999],
                    [0.9093, 0.1987, 0.0200, -0.4161, 0.9801, 0.9998],
                    [0.1411, 0.2955, 0.0300, -0.9900, 0.9553, 0.9996],
                ]
            ]
        )
        self.assertTrue(torch.allclose(embeddings, embedding_target, atol=1e-4))


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
