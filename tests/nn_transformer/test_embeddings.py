import unittest
from pl_utils.nn import (
    LearnablePosEmbedding,
    RotaryPosEmbedding,
    TimeStepSinusoidalEmbbedding,
)
import torch


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
