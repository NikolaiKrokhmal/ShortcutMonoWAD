"""Training entrypoint.  python scripts/train.py [hydra.overrides...]

pyrootutils resolves the project root and exports PROJECT_ROOT (used by configs/paths).
Imports are handled by the editable install (`pip install -e .`), so pythonpath=False.
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
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = KittiDataModule(cfg)
    model = MonoWADModule(cfg)
    logger = hydra.utils.instantiate(cfg.logger)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
