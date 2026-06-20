"""Training entrypoint.  python scripts/train.py [hydra.overrides...]

pyrootutils resolves the project root and exports PROJECT_ROOT (used by configs/paths).
Imports are handled by the editable install (`pip install -e .`), so pythonpath=False.
"""
import logging
import os
import warnings

# Quiet the cosmetic noise (deprecations, numba perf hints, PL bottleneck tips) so the
# terminal is readable. Must run BEFORE the heavy imports below — some warnings fire at
# import time. Set MONOWAD_VERBOSE=1 to restore everything (e.g. when debugging).
if not os.environ.get("MONOWAD_VERBOSE"):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="pytorch_lightning")
    try:
        from numba.core.errors import NumbaPerformanceWarning

        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    except ImportError:
        pass
    # numba's CUDA driver chatter ("init", "add pending dealloc ...") is INFO logging,
    # not warnings — silence it at the logger.
    logging.getLogger("numba").setLevel(logging.WARNING)

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
