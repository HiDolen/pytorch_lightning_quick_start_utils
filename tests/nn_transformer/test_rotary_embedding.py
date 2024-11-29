import unittest
from pl_utils.nn import get_latent_2d_ids, RotaryEmbeddingMultiDimension, apply_rotary_emb
import torch


class TestTransformerAttention(unittest.TestCase):
    def test_rope_embedding(self):
        rope_embed = RotaryEmbeddingMultiDimension(theta=10000, dim=[32, 16])
        img_pos_ids = get_latent_2d_ids(8, 8)
        freqs_cis = rope_embed(img_pos_ids)
        k_or_v = torch.randn(2, 4, 8 * 8, 32 + 16)
        k_or_v = apply_rotary_emb(k_or_v, freqs_cis)

if __name__ == '__main__':
    unittest.main()
