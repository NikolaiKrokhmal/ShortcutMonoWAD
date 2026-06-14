"""LightningDataModule for KITTI mono 3D detection.

Replaces the legacy DATASET_DICT registry + manual DataLoader construction.
Phase 2 of PLAN.md.

Receives the **full** Hydra ``cfg`` (see ``scripts/train.py``) and reads its data
config from ``cfg.data``. The train split runs the augmenting pipeline
(``build_train_transform``); val runs the deterministic one (``build_eval_transform``).
Batching is delegated to ``collate.collate_fn`` (train 7-tuple / val 6-tuple).
"""
from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .collate import collate_fn
from .dataset import KittiMonoDataset
from .transforms import build_eval_transform, build_train_transform


class KittiDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.obj_types = tuple(self.data_cfg.get("obj_types", ["Car"]))
        self.train_set: KittiMonoDataset | None = None
        self.val_set: KittiMonoDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        d = self.data_cfg
        if stage in ("fit", None) and self.train_set is None:
            self.train_set = KittiMonoDataset(
                h5_path=d.train_h5,
                transform=build_train_transform(d.rgb_mean, d.rgb_std),
                obj_types=self.obj_types,
            )
        if stage in ("fit", "validate", None) and self.val_set is None:
            self.val_set = KittiMonoDataset(
                h5_path=d.val_h5,
                transform=build_eval_transform(d.rgb_mean, d.rgb_std),
                obj_types=self.obj_types,
            )

    def train_dataloader(self) -> DataLoader:
        d = self.data_cfg
        return DataLoader(
            self.train_set,
            batch_size=d.batch_size,
            shuffle=True,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=d.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        d = self.data_cfg
        return DataLoader(
            self.val_set,
            batch_size=d.batch_size,
            shuffle=False,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=d.num_workers > 0,
        )
