# MonoWAD — `my_implementation`

Migration of MonoWAD (monocular 3D object detection) from the legacy `visualDet3D` tree
(EasyDict config, registries, hand-rolled training loop, TensorBoard) to a modern stack:
**Hydra + PyTorch Lightning + WandB**. The migration plan lives in `../PLAN.md` (4 phases).
Imports resolve as `from monowad.data.dataset import KittiMonoDataset`; intra-package
imports are **relative** (e.g. `from ..utils.geometry import ...`).

## Running things — use the Docker container

There is **no local Python env**; deps live in the `ai_ev` Docker container. The repo root
is mounted at `/workspace`, so `my_implementation` is `/workspace/my_implementation`.

The `monowad` package is **not pip-installed** (the container ships setuptools 59.6.0, too
old for PEP 660 editable installs — deferred). Put `src` on the path via `PYTHONPATH` instead:

```bash
# one-off command in the (already running) container
docker exec -w /workspace/my_implementation \
    -e PYTHONPATH=/workspace/my_implementation/src ai_ev python <script>
# launch the container (jupyter lab on :8888) if not running
./run.sh        # from repo root
```

### Docker gotchas for training (both hit during the first GPU run)

- **`--shm-size`.** Docker's default `/dev/shm` is **64 MB**; DataLoader workers pass batches
  through shared memory, so `num_workers>0` dies with `DataLoader worker exited unexpectedly`
  (worker killed by signal, *no* Python traceback). Run with `--shm-size=8g` (now in `run.sh`).
- **`num_workers>0` + CUDA = fork crash.** Lightning moves the model to GPU **before** building
  the dataloaders, so CUDA is already initialized in the parent; forked train workers inherit
  that context and abort with `CUDA error: initialization error` (`c10::cuda::ExchangeDevice`)
  at training step 0. **Workaround for now: `data.num_workers=0`** (the model/diffusion compute
  dominates, so I/O isn't the bottleneck). Proper fix (deferred): give the DataLoaders a
  `spawn`/`forkserver` `multiprocessing_context` — needs the injected transforms to be
  picklable first (the `transforms.py` factories build closures; verify before switching).
- Use an **absolute** repo path for `-v ...:/workspace` in one-off `docker run`s; `$PWD` in a
  detached/background shell may not be the repo root (mount lands wrong → "can't open train.py").

## Data: HDF5-backed pipeline

Training reads from packed `data.h5` files (one per split), **not** the legacy
`imdb.pkl` + per-file images. Built by `../scripts/pack_hdf5.py`.

- `my_implementation/data/train/data.h5` — 3712 frames (clear, foggy, depth, P2, labels)
- `my_implementation/data/val/data.h5`   — 3769 frames (clear, foggy, P2, labels; **no depth**)

**What is baked in at pack time:** `CropTop` + `Resize` only (deterministic geo ops) →
fixed `288x1280` images, plus the **post-resize `P2`**. Depth GT is stored at ¼ resolution
(`72x320`). Labels are the **raw, obj_types-filtered KITTI labels** (2D box in *original*
resolution, 3D box in camera coords), stored ragged/flattened with a per-frame `count`.

**What stays at runtime** (intentionally left out of the bake): `PhotometricDistort`,
`RandomMirror`, `Normalize`, and 3D→2D **reprojection**. Note: baking moved
PhotometricDistort to *after* resize, so training is not bit-identical to the legacy
pipeline (accepted; val/test are identical).

## Status

### Done
- `src/monowad/data/dataset.py` — `KittiMonoDataset` (HDF5-backed). Verified in-container
  against both splits + zero-label edge case.
  - Lazy, fork-safe h5 handle (`__getstate__` drops it so it's never pickled to workers).
  - Ragged-label slicing via cumulative offsets; rebuilt as attribute objects, filtered to `obj_types`.
  - Takes an **injected `transform`** callable `(clear, foggy, P2, labels, depth) -> same`
    (runs with `transform=None` for raw inspection; pipeline lives in `transforms.py`).
  - `_reproject` ported from legacy: builds `bbox3d [N,7]` (`proj_cx, proj_cy, z, w, h, l, alpha`)
    and in-frame `bbox2d [N,4]`; recomputes `alpha` through the post-mirror P2.
- `src/monowad/data/collate.py` — `collate_fn` reproduces the legacy tuple contract:
  **train → 7-tuple** (rgb, calib, labels, bbox2d, bbox3d, depth, foggy), **val → 6-tuple**
  (…, foggy), **test → 5-tuple**. Lives in its own module so `DataLoader` wiring can import it
  without the dataset class; re-exported as `KittiMonoDataset.collate_fn` for back-compat.
- `src/monowad/utils/geometry.py` — `alpha2theta_3d`, `theta2alpha_3d`, `BBox3dProjector`,
  plus `calc_iou`, `BackProjection`, `ClipBoxes` (added in Tier 2) **ported in** so the
  package no longer imports from `visualDet3D`.
- `src/monowad/data/transforms.py` — runtime augmentation pipeline (see below). Verified
  in-container against both splits.
- `src/monowad/data/datamodule.py` — `KittiDataModule` (`pl.LightningDataModule`). Takes the
  full Hydra `cfg`, reads `cfg.data`. `setup` builds the train set with `build_train_transform`
  and the val set with `build_eval_transform` (idempotent, stage-aware); `obj_types` defaults to
  `("Car",)`, overridable via `cfg.data.obj_types`. Loaders use `collate_fn` and config
  `batch_size`/`num_workers`/`pin_memory`; train shuffles + `drop_last`, val does neither;
  `persistent_workers` when workers > 0. Verified in-container (train 7-tuple, val 6-tuple).

#### Model port (Tiers 1–2) — the network itself, ported off `visualDet3D`

The model is split across `src/monowad/models/`. Intra-package imports are relative; no
file imports `visualDet3D`. Port done in two tiers (see `../PLAN.md`).

**Tier 1 — the core (framework-clean `nn.Module`s, ported near-verbatim):**
- `models/MonoWAD.py` — `MonoWAD` core: backbone → neck → weather codebook → diffusion →
  depth head → depth-aware transformer. `train` forward returns `(feat, depth, l_proposed)`,
  eval returns `(feat, depth)`. Still hardcodes `dla102(pretrained=True)` + `flash_attn=True`
  (needs GPU + network to instantiate).
- `models/dfe.py`, `dpe.py`, `dtr.py`, `wc.py` — DepthAwareFE, DepthAwarePosEnc,
  DepthAwareTransformer, WeatherCodebook (CKR loss). Leaf modules, no cross-deps.
- `models/backbone/{dla,dlaup}.py` — DLA-102 backbone + DLAUp neck.
- `models/diffusion/{denoising_diffusion_pytorch,attend}.py` — `Unet` + `GaussianDiffusion`
  (vendored lucidrains, heavily modified) + flash-attention helper.
- Verified: `from monowad.models.MonoWAD import MonoWAD` imports in-container.

**Tier 2 — the head + top-level detector:**
- `models/detector.py` — `MonoWAD_3D` (was the legacy `Detector.py`/`MonoWAD_3D`). Wires
  core → head, owns the depth loss. I/O contract unchanged: `train_forward →
  (cls_loss, reg_loss, l_proposed, loss_dict)`, `test_forward → (scores, bboxes, cls_indexes)`
  (batch 1). Registry decorator dropped; Hydra-instantiable via `_target_`. Takes a
  mapping-like `network_cfg` (`obj_types`, `mono_backbone`, `head`).
- `models/heads/detection_3d_head.py` — `AnchorBasedDetection3DHead` (assign/sample/
  encode/decode/`loss`/`get_bboxes`).
- `models/anchors.py` — `Anchors` + `generate_anchors`/`shift`.
- `models/losses.py` — `SigmoidFocalLoss`, `ModifiedSmoothL1Loss`, `bin_depths`,
  `DepthFocalLoss` (detection + depth losses consolidated).
- `models/blocks.py` — `AnchorFlatten`. `models/dcn.py` — `ModulatedDeformConvPack`.
- Verified in-container (CPU): head builds + loads anchor priors (48 anchors/loc); forward on
  `[1,256,36,160]` → `cls [1,276480,2]`, `reg [1,276480,12]`; `get_anchor` on 288×1280 →
  276480 anchors, 103968 pass the 3D frustum filter.

#### Training glue (Tier 3) — the LightningModule + Hydra detector config

- `module.py` — `MonoWADModule` (`pl.LightningModule`):
  - `training_step` inlines legacy `train_mono_detection`: builds the compound annotation,
    runs `detector([rgb, rgb.new(annotation), calib, depth, foggy])`, averages cls/reg,
    sums in `l_proposed`, logs every loss term, skips degenerate steps (`max_length==0`
    or `loss==0`). Gradient clipping is delegated to `Trainer(gradient_clip_val)`, the
    optimizer step/zero_grad to Lightning.
  - `validation_step` runs the detector's **test path** per-sample (`test_forward` needs
    batch 1) and logs `val/num_detections` — a runnable inference sanity pass.
  - `configure_optimizers` — Adam + CosineAnnealingLR via Hydra `instantiate`.
  - `_load_pretrained(ckpt_path)` — loads a detector `state_dict` (`mono_core.*` /
    `bbox_head.*`) at init with `strict=False`, printing a match report. Unwraps
    `state_dict`/`model_state_dict` wrappers and strips a leading `detector.` prefix.
    Driven by the top-level `ckpt_path` config key (null → train from scratch).
- `utils/annotations.py` — `compound_annotation` ported (packs ragged labels +
  2D/3D boxes into the dense `[B, max, 12]` target the head's `loss` consumes).
- Hydra config filled in: `configs/detector/monowad.yaml` carries the full `network_cfg`
  (obj_types / mono_backbone / head→anchors_cfg,layer_cfg,loss_cfg,test_cfg), ported from
  legacy `config/config.py`. `_target_` fixed to `MonoWAD_3D`. Optimizer/scheduler/trainer
  hyperparams aligned to legacy (`max_epochs=120`, `T_max=120`, `eta_min=5e-6`,
  `gradient_clip_val=0.1`); `clipped_gradient_norm` removed from the Adam cfg (it lives on
  the Trainer, Adam rejects unknown kwargs).
- Verified in-container (CPU): full config composes; `compound_annotation` → `[2,2,12]`
  with correct cls index + `-1` padding; the head builds straight from the YAML head block
  (48 anchors/loc); `MonoWADModule` imports.

#### Pretrained weights + GPU run

- **Loading the original MonoWAD checkpoint.** Drop a `state_dict` in
  `my_implementation/checkpoints/` (gitignored) and point `ckpt_path` at it (default:
  `${paths.root_dir}/checkpoints/MonoWAD_3D.pth`). `mono_backbone.pretrained: false` skips
  the ImageNet DLA download entirely (the checkpoint overwrites those weights anyway), so a
  warm start needs **no network access**.
- **Verified (GPU, `--network none`):** the original `MonoWAD_3D.pth` loads
  **1234/1234 tensors, 0 missing, 0 unexpected** — the port is a clean checkpoint match
  (including the torchvision-DCN param names and all diffusion buffers).
- **`fast_dev_run` passed** (GPU): model builds (66.2M params), weights load, 1 train + 1 val
  step run with no errors.
- Run a 1-epoch smoke train in the container (offline wandb, small batch for the 8 GB card):
  ```bash
  docker run --rm --gpus all -v "$PWD":/workspace -w /workspace/my_implementation \
    -e PYTHONPATH=/workspace/my_implementation/src -e WANDB_MODE=offline \
    ai_ev:latest python scripts/train.py \
      data.batch_size=2 data.num_workers=4 trainer.max_epochs=1 \
      +trainer.limit_val_batches=5 +trainer.enable_checkpointing=false
  ```
  Note: Trainer keys not already in `configs/trainer/default.yaml` need a `+` (struct mode).

### KITTI AP validation (wired) — `monowad/eval/`
- **Ported in** off `visualDet3D/evaluator/kitti`: `eval.py`, `kitti_common.py`,
  `rotate_iou.py`, `evaluate.py` (near-verbatim), plus `result_writer.py`
  (`write_result_to_file`) and `kitti_eval.py` (the `evaluate_kitti` entry + `parse_ap`).
- **GPU-only at import.** `rotate_iou.py`'s `@cuda.jit` rotated-IoU kernels compile *at
  import time* and need a CUDA device, so `kitti_eval.evaluate_kitti` imports the heavy
  modules **lazily** — importing `monowad.eval` stays CPU-safe; only scoring touches the GPU.
- **Eval flow (`module.py`):** `on_validation_epoch_start` cleans the result dir;
  `validation_step` runs `test_forward` per frame, back-projects the 3D state to camera
  coords (`BackProjection`/`BBox3dProjector`), remaps the 2D box to original resolution via
  `original_P` vs post-resize `P2` (`_remap_2d`, legacy `test_one`), and writes a KITTI
  result `.txt` named by `frame_id`; `on_validation_epoch_end` calls `evaluate_kitti` and
  `self.log()`s Car 3d/bev/bbox AP (easy/mod/hard). Controlled by `cfg.eval`
  (`enabled`, `label_dir`, `result_dir`, `score_thr`, `gpu`); set `eval.enabled=false` for
  an inference-only val pass.
- **`get_split_parts` guarded** (only deviation from the verbatim evaluator copy): the
  legacy returns leading zero-size partitions when `num_examples < num_parts` (50), crashing
  `np.concatenate([])`. It never bit legacy (full val only); guarded so partial/smoke val
  scores fine. The **full** val set (3769) takes that path natively.

### Stubs / TODO (next)
- **Full val epoch is slow.** `validation_step` pushes every val frame through the full
  15-step diffusion one at a time (~2 s/frame), so a whole-val-set AP pass is ~hours. AP
  needs the **whole** set (no `limit_val_batches`); batching the val inference path is the
  main remaining perf task.
- Containers need `--shm-size=8g` (as `run.sh` sets) when `data.num_workers>0`, else the
  val DataLoader workers can die ("worker exited unexpectedly"); `num_workers=0` also avoids it.

## Augmentation pipeline (`transforms.py`)

The dataset takes an **injected** transform with signature
`(clear, foggy, P2, labels, depth) -> same`. The pipeline is built by two factories:

- `build_train_transform(mean, std, mirror_prob=0.5, distort_prob=1.0)` →
  **ConvertToFloat → PhotometricDistort → RandomMirror → Normalize**
- `build_eval_transform(mean, std)` → **ConvertToFloat → Normalize** (deterministic)

After the transform returns, the dataset calls `_reproject` to (re)build `bbox3d`/`bbox2d`
through the (possibly mirrored) `P2` — so the transform only has to keep `P2` and the raw
KITTI label fields (`x/y/z/w/h/l/ry`, `bbox_l/t/r/b`) in lock-step with the image.

Ported from `visualDet3D/data/pipeline/stereo_augmentator.py`, pared down to what is **not**
baked into `data.h5`: `CropTop` + `Resize` (and the post-resize `P2`) are applied at pack
time, so they are absent here. The legacy `p3`/`image_gt`/`lidar` slots are dropped; `depth`
takes the old `image_gt` role (flipped on mirror, never normalized).

Pieces:
- `Compose` — chains transforms, threading the full 5-tuple.
- `ConvertToFloat`, `Normalize` — cast to float32; `/255` − mean / std on images only.
- `PhotometricDistort` (+ `RandomBrightness/Contrast/Saturation/Hue`, `ConvertColor`) —
  brightness + HSV jitter with contrast first-or-last; operates on float32 **copies**.
- `RandomMirror` — horizontal flip of clear/foggy/depth/P2/labels in lock-step.

Two deliberate departures from the legacy code (see "Key design decisions" below) are
implemented here: the **no-role-swap mirror** and **shared photometric params across
clear & foggy**. Both are covered by sanity checks (residual stays 0 when clear==foggy;
box/`x`/`cx` reflect about the width).

## Key design decisions

- **No `visualDet3D` dependency.** Geometry helpers are ported into `monowad.utils.geometry`.
- **Corrected `RandomMirror` (drop the clear/foggy swap).** The legacy `RandomMirror` lives in
  `stereo_augmentator.py` and swaps the two images (`left_image, foggy = foggy, left_image`) +
  swaps `P2/P3`. That is correct *stereo* behaviour (mirroring swaps left/right cameras), but
  MonoWAD reused the right-image slot for the foggy image, so the swap got applied to
  clear↔foggy where it has no geometric meaning — and the model is **asymmetric** (clear is the
  clean reference, `noise = foggy - clean` in the diffusion, eval uses clear only). We mirror
  clear/foggy/depth/P2/labels **in place without swapping roles**.
  - **Label implication of dropping the swap: none.** Labels are geometry-driven (flip + P2),
    identical for clear and foggy; the swap only chose which array filled which slot.
- **PhotometricDistort must share random params across clear & foggy** (sample once, apply to
  both) — the legacy primitives do this, and it keeps the weather residual coherent.

### Model-port decisions (Tier 2)

- **DCN → torchvision.** The legacy reg head uses a custom-CUDA `ModulatedDeformConvPack`
  (`visualDet3D/networks/lib/ops/dcn`) that needs a `setup.py build_ext` compile — a blocker
  in this container. Replaced by `models/dcn.py`, a thin wrapper over
  `torchvision.ops.deform_conv2d` (modulated/DCNv2 variant, CPU+CUDA, no build step). It is
  **checkpoint-compatible** with the legacy module: identical param names
  (`weight`, `bias`, `conv_offset.{weight,bias}`) and the same offset/mask convention
  (`conv_offset` emits `3·k·k` channels → split into offset_xy + sigmoid(mask)).
- **Device-safe (no hardcoded `.cuda()`).** Legacy focal loss, the empty-annotation zero
  cls-loss, and the anchor `useful_mask` all hardcoded `cuda`. Ported to device-following
  allocations (`zeros_like` / `new_zeros` / `device=self.anchors.device`) — required because
  Lightning owns device placement.
- **Flat anchor-prior path.** `Anchors` loads `<preprocessed_path>/anchor_{mean,std}_<cls>.npy`
  **directly** (the HDF5 layout puts them in `data/train/`), dropping the legacy
  `<preprocessed_path>/training/` subfolder indirection.
- **`post_optimization` not ported.** The legacy `get_bboxes` post-opt path uses
  hill-climbing + the un-ported `iou3d` CUDA op. It is off in config
  (`post_optimization=False`); enabling it raises `NotImplementedError`. This is the only
  remaining Tier-4 op, and it is optional.
- **Anchors/head config is mapping-agnostic.** The head accepts plain `dict` or OmegaConf
  `DictConfig` for `anchors_cfg`/`layer_cfg`/`loss_cfg`/`test_cfg` (both support `**` unpack and
  `getattr(cfg, k, default)`). `Anchors` coerces `ratios`/`scales`/etc. to fixed numpy/list
  types so OmegaConf `ListConfig` inputs work.

### Training-glue decisions (Tier 3)

- **`MonoWAD_3D(network_cfg=...)` via a nested Hydra block.** The detector takes a single
  mapping arg, so the YAML nests everything under `network_cfg:` (no `_target_` inside →
  Hydra leaves it a `DictConfig`). `np.array` constants from legacy (`ratios`, `scales`) are
  written as plain YAML lists; `Anchors` coerces them back. `preprocessed_path` interpolates
  `${paths.train_dir}` so anchor priors resolve at runtime.
- **Gradient clipping moved to the Trainer.** Legacy clipped by hand
  (`clip_grad_norm_(…, clipped_gradient_norm)`) inside the train loop; under Lightning this is
  `Trainer(gradient_clip_val=0.1)` and the kwarg is **removed** from the Adam cfg (Adam raises
  on unknown kwargs). Value matches legacy MonoWAD `config/config.py` (0.1, not the 35 that was
  a placeholder).
- **Loss bookkeeping is Lightning's.** `training_step` only returns the scalar loss + logs;
  `optimizer.zero_grad`/`step` and the AverageMeter/LossLogger are gone (Lightning + WandB).
  Degenerate steps return `None` (Lightning skips them) instead of the legacy early `return`.
- **Validation scores KITTI AP, not a loss.** Val has no depth GT, so the training loss can't
  be computed on it; `validation_step` runs `test_forward` and writes KITTI result files, and
  `on_validation_epoch_end` scores AP (see "KITTI AP validation" above).

## Known gaps / watch-outs

- **Eval calib comes from `P2_original` in the h5.** `pack_hdf5.py` stores both the
  *post-resize* `P2` (what the model sees) **and** `P2_original` (the pre-CropTop/Resize,
  full-KITTI-resolution calib) per frame, plus `frame_id`. The dataset returns the real
  `original_P` (falls back to post-resize `P2` only for legacy packs lacking the field —
  `dataset.has_original_P`), and the val/test `collate_fn` carries `original_P` + `frame_id`
  through (val tuple is now **8** elements; train tuple unchanged). 3D boxes are metric and
  need no remap; only the 2D box is mapped back to original resolution for 2D AP.
- **3D AP is resolution-independent**; `original_shape` is *not* needed (the legacy only used
  it in the 2D-only branch of `test_one`, which MonoWAD never takes since it runs 3D).
- **Re-pack required** for the above: `data.h5` written before `P2_original` was added lacks
  it. After re-packing (writes to `workdirs/MonoWAD/output/{training,validation}`), copy the
  files to `my_implementation/data/{train,val}/data.h5` (where `paths.{train,val}_dir` point).
