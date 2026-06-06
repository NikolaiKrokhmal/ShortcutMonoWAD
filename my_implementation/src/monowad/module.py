"""MonoWADModule — the LightningModule tying detector + optim + eval together.

Inlines the logic that used to live in visualDet3D/networks/pipelines/{trainers,evaluators,testers}.py.
Phase 3 of PLAN.md.
"""
from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


class MonoWADModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.detector = hydra.utils.instantiate(cfg.detector)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        # TODO: run monowad.eval.kitti_eval and self.log() the AP metrics
        pass

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
