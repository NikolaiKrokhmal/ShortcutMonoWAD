# MonoWAD (reimplementation)

Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection.
Reimplementation on a modern stack: **Hydra** (config) + **PyTorch Lightning** (training)
+ **WandB** (logging). See `../PLAN.md` for the migration plan from the legacy `visualDet3D` code.

## Setup

```bash
pip install -e .          # installs the `monowad` package (editable)
```

Imports then resolve cleanly from anywhere:

```python
from monowad.data.datamodule import KittiDataModule
from monowad.module import MonoWADModule
```

## Layout

| Path | What |
|------|------|
| `configs/`        | Hydra config tree (composed via `config.yaml`) |
| `src/monowad/`    | the package — data, models, LightningModule, eval, utils |
| `scripts/`        | thin entrypoints (`pyrootutils` + `@hydra.main`) |
| `notebooks/`      | exploration |
| `data/`           | gitignored — KITTI + preprocessed artifacts |
| `outputs/`        | gitignored — Hydra runs, checkpoints, wandb |

## Run

```bash
python scripts/train.py                      # default config
python scripts/train.py trainer.fast_dev_run=true
python scripts/train.py data.batch_size=16 optimizer.lr=1e-4
```
