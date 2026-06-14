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

## Training

`scripts/train.py` is the single entrypoint. `pyrootutils` resolves the project root,
Hydra composes `configs/config.yaml`, and PyTorch Lightning drives `trainer.fit`. Any config
value is overridable on the CLI as `group.key=value` (use `+group.key=value` to *add* a key
that isn't already in the YAML — Hydra runs in struct mode).

### Running in the container

There is **no local Python env** — deps live in the `ai_ev` Docker image, and the package is
not pip-installed, so `src` goes on `PYTHONPATH`. A full single-run invocation:

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace/my_implementation \
  -e PYTHONPATH=/workspace/my_implementation/src -e WANDB_MODE=offline \
  ai_ev:latest python scripts/train.py \
    data.batch_size=2 data.num_workers=0 trainer.max_epochs=120
```

Quick sanity pass (builds the model, loads weights, runs 1 train + 1 val step, exits):

```bash
... python scripts/train.py trainer.fast_dev_run=true
```

If the container is already running (`./run.sh` from the repo root), use `docker exec` instead:

```bash
docker exec -w /workspace/my_implementation \
  -e PYTHONPATH=/workspace/my_implementation/src ai_ev \
  python scripts/train.py <overrides...>
```

Bare `python scripts/train.py <overrides...>` works too, *if* you have an env where the
`monowad` package imports.

### Options worth setting & why

| Override | Default | Why you'd change it |
|----------|---------|---------------------|
| `data.batch_size=N` | 8 | Drop to `2` on an 8 GB card — the diffusion U-Net is memory-heavy. |
| `data.num_workers=N` | 8 | **Set `0` on GPU.** Lightning inits CUDA in the parent before building loaders; forked workers then crash at step 0 (`CUDA error: initialization error`). Compute dominates I/O, so 0 costs little. Needs `--shm-size=8g` for `>0` regardless (see CLAUDE.md). |
| `trainer.max_epochs=N` | 120 | Length of the run. Mirror the cosine schedule with `scheduler.T_max=N` so the LR actually anneals over the full run. |
| `ckpt_path=...` | `checkpoints/MonoWAD_3D.pth` | Warm-start from a MonoWAD `state_dict` (loaded `strict=False`). Set `ckpt_path=null` to train from scratch. |
| `detector.network_cfg.mono_backbone.pretrained=true` | false | Pull ImageNet DLA weights on init (one-time download). Leave `false` when warm-starting — the checkpoint overwrites them anyway, and `false` means **no network access** is needed. |
| `optimizer.lr=1e-4` / `optimizer.weight_decay=0.0` | 1e-4 / 0 | Adam hyperparams (mirror legacy). |
| `scheduler.eta_min=5e-6` | 5e-6 | Cosine LR floor. |
| `trainer.gradient_clip_val=0.1` | 0.1 | Gradient-norm clip (lives on the Trainer, not the optimizer). |
| `trainer.check_val_every_n_epoch=5` | 5 | Validation cadence. |
| `trainer.precision=16` | 32 | Mixed precision to save memory. |
| `+trainer.limit_val_batches=N` | — | **Cap validation** for smoke runs. Each val frame runs the full 15-step diffusion one at a time (~2 s/frame), so a whole-val pass is hours. NB: a real AP number needs the *whole* val set — don't cap it when you actually want a score. |
| `+trainer.enable_checkpointing=false` | true | Skip checkpoint writing for smoke runs. |
| `eval.enabled=false` | true | Skip KITTI AP scoring → inference-only val (faster sanity pass). |
| `eval.label_dir=...` | `${paths.root_dir}/../data/KITTI/object/training/label_2` | KITTI ground-truth `label_2` dir for AP. Override if your KITTI tree is elsewhere. |
| `eval.score_thr=0.4` | 0.4 | Min confidence for a detection to be written to its KITTI result file. |
| `trainer.fast_dev_run=true` | false | One train + one val batch, no logging/checkpoints — fastest "does it run" check. |
| `seed=42` | 42 | Global seed (`seed_everything(..., workers=True)`). |

### Where outputs land

Hydra creates a timestamped run dir under `outputs/` (gitignored) for each invocation —
logs, config snapshot, and (unless disabled) Lightning checkpoints.

## Viewing results in WandB

Logging is configured by `configs/logger/wandb.yaml` (`WandbLogger`, project **`monowad`**,
`save_dir=outputs/`). The module logs, per step:

- `train/loss`, `train/cls_loss`, `train/reg_loss`, `train/proposed_loss`, plus each
  individual loss term from the detector's loss dict;
- `val/num_detections` — the per-frame detection count from the inference pass;
- KITTI AP (when `eval.enabled=true`, on the validation cadence): per class
  `val/<cls>_{3d,bev,bbox}_{easy,mod,hard}` — e.g. `val/Car_3d_mod` (the headline number,
  also on the progress bar). The full per-class AP table is printed to stdout too. See
  CLAUDE.md "KITTI AP validation" for the flow.

**Online (default):** runs land at `https://wandb.ai/<your-entity>/monowad`. Log in once with
`wandb login` (or set `WANDB_API_KEY`). Set the entity via `logger.entity=<team-or-user>` and a
readable run name with `logger.name=<name>` (else auto-generated). Open the project URL printed
at startup to watch loss curves live.

**Offline (no network / inside the container):** set `WANDB_MODE=offline` (as in the commands
above). Runs are written to `outputs/wandb/offline-run-*`. Push them to the web UI later with:

```bash
wandb sync outputs/wandb/offline-run-*
```
