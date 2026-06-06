"""Evaluation entrypoint.  python scripts/eval.py ckpt_path=...

Loads a trained checkpoint and runs validation / KITTI AP evaluation.
"""
import pyrootutils

root = pyrootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=False, dotenv=True
)

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from monowad.data.datamodule import KittiDataModule
from monowad.module import MonoWADModule


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = KittiDataModule(cfg)
    model = MonoWADModule.load_from_checkpoint(cfg.ckpt_path, cfg=cfg)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
