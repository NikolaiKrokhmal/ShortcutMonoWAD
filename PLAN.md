# Migration Plan: Hydra + PyTorch Lightning + WandB

## Overview

Migrate MonoWAD from its current manual training infrastructure (EasyDict config, TensorBoard,
hand-rolled training loop, registry pattern) to a modern stack:

- **Hydra** — YAML-based config with composition and CLI overrides
- **PyTorch Lightning** — structured training loop, DDP, checkpointing
- **WandB** — experiment logging and metric tracking

The existing `DETECTOR_DICT` / `DATASET_DICT` / `PIPELINE_DICT` registry pattern will be
**removed** — Hydra's `instantiate()` replaces class registries, and Lightning replaces the
pipeline function registry entirely.

Migration is broken into 4 phases, each ending with a runnable verification step.

---

## Phase 1 — Preprocessing & Data Exploration

**Goal:** Produce preprocessed artifacts and understand the data format before touching model code.

### Tasks
1. Run preprocessing scripts inside Docker:
   - `scripts/imdb_precompute_3d.py` → `workdirs/MonoWAD/output/training/imdb.pkl`,
     `anchor_mean_Car.npy`, `anchor_std_Car.npy`
   - `scripts/depth_gt_compute.py` → `workdirs/MonoWAD/output/training/depth/P2*.png`
   - Re-run for validation split → `output/validation/imdb.pkl`

2. Explore in a Jupyter notebook:
   - Load `imdb.pkl`, inspect `KittiData` frame objects
   - Manually instantiate `KittiMonoDataset`, call `__getitem__` on a few indices
   - Visualize: image, foggy image, projected 3D boxes, depth map
   - Print `collate_fn` output shapes and dtypes

### Done when
- Notebook runs without error
- Image + annotation alignment looks correct visually
- Output tensor shapes from `collate_fn` are documented

---

## Phase 2 — Dataset / DataLoader / LightningDataModule

**Goal:** Replace registry-based dataset instantiation with a clean `LightningDataModule`
driven by a minimal Hydra config.

### Tasks

1. Create minimal Hydra configs (paths + data only):
   ```
   configs/
     config.yaml
     paths/default.yaml     # data_path, preprocessed_path, visualDet3D_path
     data/kitti.yaml        # batch_size, num_workers, rgb_shape, augmentation,
                            # train_split_file, val_split_file
   ```
   NumPy arrays (rgb_mean, rgb_std) become plain lists in YAML; convert to np.array at use sites.

2. Refactor `visualDet3D/data/kitti/dataset/mono_dataset.py`:
   - Remove `@DATASET_DICT.register_module` decorator and its import
   - OmegaConf `DictConfig` supports attribute access, so internal field access is unchanged

3. Create `visualDet3D/data/lightning_datamodule.py`:
   ```python
   class KittiDataModule(pl.LightningDataModule):
       def setup(self, stage):   # instantiate KittiMonoDataset directly
       def train_dataloader(self): ...
       def val_dataloader(self):   ...
   ```

### Done when
```python
cfg = OmegaConf.load("configs/config.yaml")
dm = KittiDataModule(cfg)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
# shapes and dtypes match Phase 1 notebook
```

---

## Phase 3 — Model + LightningModule

**Goal:** Wrap the detector in a `LightningModule`, remove remaining registries.

### Tasks

1. Extend Hydra configs with remaining groups:
   ```
   configs/
     optimizer/adam.yaml      # type_name, lr, weight_decay, clipped_gradient_norm
     scheduler/cosine.yaml    # type_name, T_max, eta_min
     detector/monowad.yaml    # head, anchors, loss, test cfg + _target_ for instantiate()
     trainer/default.yaml     # max_epochs, save_iter, test_iter, random_seed
     logger/wandb.yaml        # project, entity, run_name
   ```

2. Remove registries:
   - `visualDet3D/networks/detectors/__init__.py` — drop `@DETECTOR_DICT.register_module`
   - `visualDet3D/networks/utils/registry.py` — delete once all callers removed

3. Create `visualDet3D/networks/lightning_module.py`:
   ```python
   class MonoWADModule(pl.LightningModule):
       def __init__(self, cfg):
           self.detector = hydra.utils.instantiate(cfg.detector)

       def training_step(self, batch, batch_idx):
           # loss logic inlined from trainers.py; use self.log() for metrics

       def validation_step(self, batch, batch_idx):
           # per-sample inference; accumulate predictions

       def on_validation_epoch_end(self):
           # call visualDet3D/evaluator/kitti/evaluate.py; self.log() AP

       def configure_optimizers(self):
           # reuse optimizers.build_optimizer + schedulers.build_scheduler
   ```

4. Delete pipeline files once logic is inlined:
   - `visualDet3D/networks/pipelines/trainers.py`
   - `visualDet3D/networks/pipelines/evaluators.py`
   - `visualDet3D/networks/pipelines/testers.py`

### Done when
```python
module = MonoWADModule(cfg)
loss = module.training_step(batch, 0)
assert loss.item() > 0
```

---

## Phase 4 — Training Script + WandB

**Goal:** Replace the manual training loop with Lightning `Trainer` + WandB logger.

### Tasks

1. Rename `scripts/train.py` → `scripts/train_legacy.py` (keep for reference)

2. Rewrite `scripts/train.py`:
   ```python
   @hydra.main(config_path="../configs", config_name="config", version_base=None)
   def main(cfg: DictConfig):
       setup_paths(cfg)
       set_random_seed(cfg.trainer.random_seed)

       trainer = pl.Trainer(
           max_epochs=cfg.trainer.max_epochs,
           accelerator="gpu", devices=1,
           logger=WandbLogger(project=cfg.logger.project, ...),
           callbacks=[ModelCheckpoint(...), LearningRateMonitor(...)],
           gradient_clip_val=cfg.optimizer.clipped_gradient_norm,
           check_val_every_n_epoch=cfg.trainer.test_iter,
       )
       trainer.fit(MonoWADModule(cfg), datamodule=KittiDataModule(cfg))
   ```

3. Update `train.sh` to call `python scripts/train.py`

4. Keep `config/config.py` unchanged — preprocessing scripts still depend on it

### Done when
- `trainer.fast_dev_run=true` completes without error
- WandB run visible in dashboard with loss scalars
- `.ckpt` checkpoint saved under `cfg.paths.checkpoint_path`

---

## Files Changed

| File | Action |
|------|--------|
| `configs/**` | Create (Phases 2–4) |
| `visualDet3D/data/lightning_datamodule.py` | Create (Phase 2) |
| `visualDet3D/data/kitti/dataset/mono_dataset.py` | Edit: remove registry decorator (Phase 2) |
| `visualDet3D/networks/lightning_module.py` | Create (Phase 3) |
| `visualDet3D/networks/detectors/__init__.py` | Edit: remove registry decorator (Phase 3) |
| `visualDet3D/networks/utils/registry.py` | Delete (Phase 3) |
| `visualDet3D/networks/pipelines/trainers.py` | Delete (Phase 3) |
| `visualDet3D/networks/pipelines/evaluators.py` | Delete (Phase 3) |
| `visualDet3D/networks/pipelines/testers.py` | Delete (Phase 3) |
| `scripts/train.py` | Rewrite (Phase 4); old → `train_legacy.py` |
| `train.sh` | Update (Phase 4) |
| `config/config.py` | Keep unchanged |
