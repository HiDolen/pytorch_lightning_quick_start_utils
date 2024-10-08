import unittest
from pl_utils.nn import GeGLUMLP, LearnablePosEmbedding, RotaryPosEmbedding
import torch


class TestGeGLUMLP(unittest.TestCase):
    def test_forward(self):
        mlp = GeGLUMLP(10, 40)
        x = torch.randn(32, 10)
        y = mlp(x)
        self.assertEqual(y.shape, (32, 10))

class TestLearnablePosEmbedding(unittest.TestCase):
    def test_forward(self):
        pos_emb = LearnablePosEmbedding(10, 100)
        x = torch.randn(32, 100, 10)
        y = pos_emb(x)
        self.assertEqual(y.shape, (32, 100, 10))

class TestRotaryPosEmbedding(unittest.TestCase):
    def test_forward(self):
        pos_emb = RotaryPosEmbedding(10, 100)
        q = torch.randn(32, 4, 100, 10)
        k = torch.randn(32, 4, 100, 10)
        q, k = pos_emb(q, k)
        self.assertEqual(q.shape, (32, 4, 100, 10))
        self.assertEqual(k.shape, (32, 4, 100, 10))

if __name__ == '__main__':
    unittest.main()
