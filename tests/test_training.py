import unittest
import torch
from torch.utils.data import DataLoader
from torch import nn
import lightning as L
from pl_utils import LearningRateConfig, TrainingConfig
from pl_utils import BaseModule


class TestTraningProcess(unittest.TestCase):
    def test_training_process(self):
        lr_config = LearningRateConfig(
            lr_initial=1e-3,
            lr_max=1e-2,
            lr_end=1e-4,
            lr_warmup_steps=0.1,
            lr_cycle_steps=0.9,
            max_steps=100,
        )
        training_config = TrainingConfig(optimizer="adamw")
        dummy_dataset = torch.randn(32, 10)
        dummy_loader = DataLoader(dummy_dataset, batch_size=16)
        dummy_model = nn.LazyLinear(1)

        class DummyModule(BaseModule):
            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                y = self(batch)
                loss = y.mean()
                return loss

        pl_model = DummyModule(dummy_model, lr_config, training_config)
        trainer = L.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(pl_model, dummy_loader)


if __name__ == '__main__':
    unittest.main()
