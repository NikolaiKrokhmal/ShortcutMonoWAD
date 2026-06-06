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
- `src/monowad/utils/geometry.py` — `alpha2theta_3d`, `theta2alpha_3d`, `BBox3dProjector`
  **ported in** so the package no longer imports from `visualDet3D`.
- `src/monowad/data/transforms.py` — runtime augmentation pipeline (see below). Verified
  in-container against both splits.
- `src/monowad/data/datamodule.py` — `KittiDataModule` (`pl.LightningDataModule`). Takes the
  full Hydra `cfg`, reads `cfg.data`. `setup` builds the train set with `build_train_transform`
  and the val set with `build_eval_transform` (idempotent, stage-aware); `obj_types` defaults to
  `("Car",)`, overridable via `cfg.data.obj_types`. Loaders use `collate_fn` and config
  `batch_size`/`num_workers`/`pin_memory`; train shuffles + `drop_last`, val does neither;
  `persistent_workers` when workers > 0. Verified in-container (train 7-tuple, val 6-tuple).

### Stubs / TODO
- `src/monowad/models/*`, `module.py` — stubs (Phase 3).

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

## Known gaps / watch-outs

- **Eval-time calib.** The h5 stores only *post-resize* `P2` (no original-resolution P2 or
  original image size). KITTI 2D-AP eval needs to map boxes back to original resolution —
  recover via `frame_id` + KITTI calib files at eval time (Phase 3 concern).
- `dataset.original_P` / `original_shape` are the **post-resize** values (best available here),
  not the true pre-resize originals the legacy dataset carried.
