"""Pack the precomputed KITTI inputs of a split into a single HDF5 container.

Motivation
----------
At train time every sample currently opens three files (origin_2, foggy_2 and the
precomputed depth map). On a shared HPC filesystem the thousands of tiny files cost
far more in metadata/inode pressure and transfer time than the bytes themselves.
This collapses a whole split into one ``data.h5``.

What is baked in
----------------
``CropTop`` and ``Resize`` are deterministic (the only spatial randomness in the
pipeline is ``RandomMirror``), so we apply them *here*, once, and store fixed-size
``288x1280`` images. The post-resize ``P2`` is stored alongside, because the runtime
pipeline -- with ``CropTop``/``Resize`` removed -- still needs it for reprojection,
anchor matching and mirroring. The remaining ops (``PhotometricDistort``,
``RandomMirror``, ``Normalize``) stay at runtime.

Note: baking moves ``PhotometricDistort`` to *after* the resize for the training
split, so training is not bit-identical to the unbaked pipeline (validation/test have
no distort and are exactly identical). This is intentional and was accepted.

Indexing
--------
We iterate ``imdb.pkl`` directly, so row ``i`` of every dataset corresponds to
``imdb[i]`` -- the same integer index the dataset already uses, including for the
depth map (``depth/P2{i:06d}.png``).

Datasets written to ``{preprocessed_path}/{split}/data.h5``:
    clear         uint8   [N, 288, 1280, 3]  baked origin_2 (RGB)
    foggy         uint8   [N, 288, 1280, 3]  baked foggy_2  (training/validation only)
    depth         uint16  [N, Hd, Wd]        precomputed depth GT (training only)
    P2            float32 [N, 3, 4]          post-resize calibration
    P2_original   float32 [N, 3, 4]          original-resolution calibration (pre CropTop/Resize)
    frame_id      str     [N]                original KITTI id (sanity/debug)
    labels/count  int32   [N]               #objects per frame (training/validation only)
    labels/data   float32 [M, 14]           flattened label rows, cols = ``label_fields`` attr
    labels/type   str     [M]               flattened per-object class name

Labels are ragged, so they are stored flattened across all frames: frame ``i`` owns
the rows ``labels/data[offset:offset+count[i]]`` where ``offset = sum(count[:i])`` (and
likewise for ``labels/type``). These are the *raw* obj_types-filtered KITTI labels (2D
bbox in original resolution, NOT the baked frame) -- exactly what ``imdb.pkl`` holds --
so a consumer reprojects 3D boxes through the stored post-resize ``P2`` at runtime, the
same way the file-based pipeline does.

Usage
-----
    python scripts/pack_hdf5.py --config=config/config.py --split=training
    python scripts/pack_hdf5.py --config=config/config.py --split=validation
    python scripts/pack_hdf5.py --config=config/config.py --split=all
    python scripts/pack_hdf5.py --config=config/config.py --split=training --compression=none
"""
import os
import pickle

import numpy as np
import cv2
from tqdm import tqdm
from fire import Fire

from _path_init import *
from visualDet3D.data.kitti.utils import read_image
from visualDet3D.data.pipeline.augmentation_builder import Compose
from visualDet3D.utils.utils import cfg_from_file

try:
    import h5py
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError("pack_hdf5.py requires h5py (`pip install h5py`).") from exc


# Per-split contents and which raw dir / augmentation list to read from.
SPLIT_SPEC = {
    "training":   dict(aug_key="train_augmentation", foggy=True,  depth=True,  label=True),
    "validation": dict(aug_key="test_augmentation",  foggy=True,  depth=False, label=True),
    "test":       dict(aug_key="test_augmentation",  foggy=False, depth=False, label=False),
}

# Deterministic geometric ops we bake at pack time (in pipeline order).
GEO_OPS = ("CropTop", "Resize")

# Numeric fields of a KITTI label row, in file order (the leading class string is
# stored separately). The 2D bbox (bbox_[l,t,r,b]) is in *original* image resolution
# -- it is NOT the baked 288x1280 frame -- because, like imdb.pkl, we store the raw
# (obj_types-filtered) labels and leave geometry/reprojection to the runtime. The
# stored P2 is post-resize, so a consumer reprojects 3D boxes through it just as the
# file-based pipeline does.
LABEL_FIELDS = ("truncated", "occluded", "alpha",
                "bbox_l", "bbox_t", "bbox_r", "bbox_b",
                "h", "w", "l", "x", "y", "z", "ry")


def label_row(obj):
    """KittiObj -> float32 vector following LABEL_FIELDS."""
    return np.array([getattr(obj, f) for f in LABEL_FIELDS], dtype=np.float32)


def build_geo_pipeline(cfg, aug_key):
    """A Compose of just the deterministic geometric ops, taken from the config so
    it stays in sync with whatever CropTop/Resize settings the model trains with."""
    geo_cfg = [a for a in cfg.data[aug_key] if a.type_name in GEO_OPS]
    print("  baking from {}: {}".format(aug_key, [a.type_name for a in geo_cfg]))
    return Compose(geo_cfg, is_return_all=True)


def pack_split(cfg, split, compression="lzf"):
    if split not in SPLIT_SPEC:
        raise ValueError("unknown split {!r}, expected one of {}".format(split, list(SPLIT_SPEC)))
    spec = SPLIT_SPEC[split]
    pre = cfg.path.preprocessed_path

    imdb_path = os.path.join(pre, split, "imdb.pkl")
    if not os.path.isfile(imdb_path):
        raise FileNotFoundError("{} not found - run det_precompute for '{}' first".format(imdb_path, split))
    with open(imdb_path, "rb") as f:
        imdb = pickle.load(f)
    n = len(imdb)
    print("[{}] {} frames".format(split, n))

    geo = build_geo_pipeline(cfg, spec["aug_key"])
    depth_dir = os.path.join(pre, "training", "depth")
    out_path = os.path.join(pre, split, "data.h5")

    with h5py.File(out_path, "w") as h5:
        # Datasets that need an image shape are created lazily on the first frame.
        clear_ds = foggy_ds = depth_ds = None
        p2_ds = h5.create_dataset("P2", shape=(n, 3, 4), dtype="float32")
        p2_orig_ds = h5.create_dataset("P2_original", shape=(n, 3, 4), dtype="float32")
        fid_ds = h5.create_dataset("frame_id", shape=(n,), dtype=h5py.string_dtype("utf-8"))

        # Labels are ragged (variable #objects/frame). We store them flattened across
        # all frames plus a per-frame count, so frame i owns rows
        # [cumsum(count)[:i].sum() : +count[i]]. Accumulated here, written after the loop.
        label_counts = []        # [N]   objects per frame
        label_rows = []          # [M, 14] flattened numeric fields (M = total objects)
        label_types = []         # [M]   flattened class strings

        for i, frame in enumerate(tqdm(imdb, desc=split)):
            clear = read_image(frame.origin2_path)
            foggy = read_image(frame.foggy2_path) if spec["foggy"] else None
            p2 = np.asarray(frame.calib.P2, dtype=np.float32).copy()
            p2_original = p2.copy()  # snapshot before geo() mutates it (post-resize -> p2_b)

            # Bake CropTop + Resize; geo also returns the adjusted P2.
            clear_b, foggy_b, p2_b = geo(clear, foggy, p2=p2)[:3]

            if clear_ds is None:
                h, w = clear_b.shape[:2]
                clear_ds = h5.create_dataset(
                    "clear", shape=(n, h, w, 3), dtype="uint8",
                    chunks=(1, h, w, 3), compression=compression)
                if spec["foggy"]:
                    foggy_ds = h5.create_dataset(
                        "foggy", shape=(n, h, w, 3), dtype="uint8",
                        chunks=(1, h, w, 3), compression=compression)

            clear_ds[i] = np.ascontiguousarray(clear_b, dtype=np.uint8)
            if spec["foggy"]:
                foggy_ds[i] = np.ascontiguousarray(foggy_b, dtype=np.uint8)
            p2_ds[i] = p2_b
            p2_orig_ds[i] = p2_original
            fid_ds[i] = os.path.splitext(os.path.basename(frame.origin2_path))[0]

            if spec["label"]:
                objs = frame.label or []  # already filtered to cfg.obj_types in det_precompute
                label_counts.append(len(objs))
                for obj in objs:
                    label_rows.append(label_row(obj))
                    label_types.append(obj.type)

            if spec["depth"]:
                dpath = os.path.join(depth_dir, "P2%06d.png" % i)
                depth = cv2.imread(dpath, -1)  # -1: keep 16-bit single channel
                if depth is None:
                    raise FileNotFoundError("{} missing - run depth_gt_compute.py first".format(dpath))
                if depth_ds is None:
                    hd, wd = depth.shape[:2]
                    depth_ds = h5.create_dataset(
                        "depth", shape=(n, hd, wd), dtype=depth.dtype,
                        chunks=(1, hd, wd), compression=compression)
                depth_ds[i] = depth

        if spec["label"]:
            m = len(label_rows)
            data = np.stack(label_rows) if m else np.zeros((0, len(LABEL_FIELDS)), np.float32)
            grp = h5.create_group("labels")
            grp.create_dataset("count", data=np.asarray(label_counts, dtype=np.int32))
            grp.create_dataset("data", data=data.astype(np.float32),
                               chunks=True if m else None, compression=compression if m else None)
            grp.create_dataset("type", shape=(m,), dtype=h5py.string_dtype("utf-8"),
                               data=np.array(label_types, dtype=object))
            print("  labels: {} objects across {} frames".format(m, n))

        h5.attrs["split"] = split
        h5.attrs["count"] = n
        h5.attrs["baked_ops"] = ",".join(GEO_OPS)
        h5.attrs["has_foggy"] = spec["foggy"]
        h5.attrs["has_depth"] = spec["depth"]
        h5.attrs["has_label"] = spec["label"]
        if spec["label"]:
            h5.attrs["label_fields"] = ",".join(LABEL_FIELDS)

    size_gb = os.path.getsize(out_path) / 1e9
    print("[{}] wrote {} ({:.2f} GB)".format(split, out_path, size_gb))
    return out_path


def main(config="config/config.py", split="training", compression="lzf"):
    """Pack one split (or 'all' = training + validation) into an HDF5 container.

    config(str):      path to the config file.
    split(str):       'training', 'validation', 'test', or 'all'.
    compression(str): 'lzf' (fast, default), 'gzip', or 'none' for uncompressed.
    """
    cfg = cfg_from_file(config)
    if isinstance(compression, str) and compression.lower() in ("none", "", "false"):
        compression = None

    splits = ("training", "validation") if split == "all" else (split,)
    for s in splits:
        pack_split(cfg, s, compression=compression)
    print("packing finished")


if __name__ == "__main__":
    Fire(main)
