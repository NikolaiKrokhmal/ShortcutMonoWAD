"""KittiMonoDataset — HDF5-backed port of visualDet3D/data/kitti/dataset/mono_dataset.py.

Differences from the legacy file-based dataset
-----------------------------------------------
* Reads from a single packed ``data.h5`` (see ``scripts/pack_hdf5.py``) instead of
  unpickling ``imdb.pkl`` + opening three files per sample. ``CropTop`` and ``Resize``
  are already baked into the stored 288x1280 ``clear``/``foggy`` images and the stored
  post-resize ``P2``; only the non-geometric / mirror augmentations run here.
* The ``@DATASET_DICT.register_module`` registry decorator is dropped — instantiated
  directly (or via Hydra).
* The augmentation pipeline is *injected* as a ``transform`` callable rather than built
  internally, so the dataset can be exercised with ``transform=None`` for raw inspection
  and the train/val pipelines live in ``transforms.py``.

What still happens at runtime (matching the legacy ``__getitem__``)
-------------------------------------------------------------------
1. grab ``clear`` / ``foggy`` / ``P2`` / ``depth`` and the frame's labels from the h5
2. ``transform(clear, foggy, P2, labels, depth)`` — photometric distort, corrected
   mirror (no clear/foggy swap), normalize. The transform must flip ``depth``/``P2``/
   labels in lock-step with the image on mirror.
3. ``_reproject`` — project the raw KITTI 3D boxes through the (possibly mirrored) P2 to
   build the ``bbox3d`` ``[N, 7]`` regression target and the in-frame ``bbox2d`` ``[N, 4]``.

The stored labels are the *raw*, obj_types-filtered KITTI labels (2D box in original
resolution, 3D box in camera coords) — exactly what ``imdb.pkl`` held — so reprojection
through the stored post-resize P2 is mandatory, just as in the file-based pipeline.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.geometry import BBox3dProjector, theta2alpha_3d
from .collate import collate_fn

# Signature of an injected augmentation pipeline.
Transform = Callable[
    [np.ndarray, Optional[np.ndarray], np.ndarray, List[SimpleNamespace], Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[SimpleNamespace], Optional[np.ndarray]],
]


class KittiMonoDataset(Dataset):
    """KITTI mono 3D detection samples read from a packed ``data.h5``."""

    def __init__(
        self,
        h5_path: str,
        transform: Optional[Transform] = None,
        obj_types: Sequence[str] = ("Car",),
        is_reproject: bool = True,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.transform = transform
        self.obj_types = tuple(obj_types)
        self.is_reproject = is_reproject
        self.projector = BBox3dProjector()

        # Open once to read metadata + build the ragged-label index, then close. The
        # data handle is (re)opened lazily per worker in __getitem__ (see _file).
        self._h5: Optional[h5py.File] = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f.attrs["count"])
            self.has_foggy = bool(f.attrs["has_foggy"])
            self.has_depth = bool(f.attrs["has_depth"])
            self.has_label = bool(f.attrs["has_label"])
            self.is_train = self.has_depth  # only the training split carries depth GT
            # ``P2_original`` (full-res calib, for eval-time 2D remap) is only present in
            # h5 files packed after that field was added; older packs lack it.
            self.has_original_P = "P2_original" in f
            if self.has_label:
                self.label_fields = f.attrs["label_fields"].split(",")
                counts = f["labels/count"][:].astype(np.int64)
                # frame i owns label rows [offsets[i] : offsets[i + 1]]
                self.label_offsets = np.concatenate([[0], np.cumsum(counts)])
            else:
                self.label_fields = []
                self.label_offsets = np.zeros(self.length + 1, dtype=np.int64)

    # ------------------------------------------------------------------ helpers
    def _file(self) -> h5py.File:
        """Lazily (re)open the h5 handle. h5py handles are not fork-safe, so each
        DataLoader worker opens its own on first access."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self) -> dict:
        # Never pickle an open h5 handle across the fork to worker processes.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _labels(self, h5: h5py.File, index: int) -> List[SimpleNamespace]:
        """Slice this frame's labels out of the flattened ragged arrays and rebuild
        lightweight objects with attribute access (a KittiObj stand-in)."""
        if not self.has_label:
            return []
        start, end = int(self.label_offsets[index]), int(self.label_offsets[index + 1])
        if end == start:
            return []
        rows = h5["labels/data"][start:end]
        types = h5["labels/type"][start:end]
        labels: List[SimpleNamespace] = []
        for t, row in zip(types, rows):
            t = t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t)
            if t not in self.obj_types:  # already filtered at pack time; belt-and-braces
                continue
            obj = SimpleNamespace(type=t, **{f: float(v) for f, v in zip(self.label_fields, row)})
            labels.append(obj)
        return labels

    def _reproject(
        self, P2: np.ndarray, labels: List[SimpleNamespace]
    ) -> Tuple[List[SimpleNamespace], np.ndarray]:
        """Port of the legacy ``_reproject``: project each 3D box through P2 to get the
        ``[N, 7]`` state ``[proj_cx, proj_cy, z, w, h, l, alpha]`` and, when
        ``is_reproject``, overwrite the 2D box with the tight envelope of the projected
        3D corners. ``alpha`` is recomputed from ``ry`` through the given (post-mirror) P2."""
        bbox3d_state = np.zeros([len(labels), 7], dtype=np.float32)
        for obj in labels:
            obj.alpha = theta2alpha_3d(obj.ry, obj.x, obj.z, P2)

        bbox3d_origin = torch.tensor(
            [[o.x, o.y - 0.5 * o.h, o.z, o.w, o.h, o.l, o.alpha] for o in labels],
            dtype=torch.float32,
        )
        _, homo_corner, _ = self.projector(bbox3d_origin, torch.as_tensor(P2, dtype=torch.float32))

        for i, obj in enumerate(labels):
            extended_center = np.array([obj.x, obj.y - 0.5 * obj.h, obj.z, 1.0])[:, np.newaxis]
            image_center = (P2 @ extended_center)[:, 0]  # [3]
            image_center[0:2] /= image_center[2]
            bbox3d_state[i] = np.concatenate([image_center, [obj.w, obj.h, obj.l, obj.alpha]])

        max_xy, _ = homo_corner[:, :, 0:2].max(dim=1)  # [N, 2]
        min_xy, _ = homo_corner[:, :, 0:2].min(dim=1)  # [N, 2]
        bbox2d = torch.cat([min_xy, max_xy], dim=-1).cpu().numpy()  # [N, 4] x1,y1,x2,y2

        if self.is_reproject:
            for i, obj in enumerate(labels):
                obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b = bbox2d[i]

        return labels, bbox3d_state

    # ------------------------------------------------------------------ dataset API
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        h5 = self._file()
        index = index % self.length

        clear = h5["clear"][index]  # uint8 [H, W, 3]
        foggy = h5["foggy"][index] if self.has_foggy else None
        P2 = h5["P2"][index].astype(np.float32)
        depth = h5["depth"][index] if self.has_depth else None
        labels = self._labels(h5, index)

        original_shape = clear.shape
        # ``original_P`` is the *pre*-CropTop/Resize calibration (full KITTI resolution),
        # stored separately at pack time and left untransformed (eval has no mirror). The
        # eval-time 2D-box remap uses it to map predictions back to original resolution.
        # Falls back to the post-resize P2 for legacy packs that predate the field.
        if self.has_original_P:
            original_P = h5["P2_original"][index].astype(np.float32)
        else:
            original_P = P2.copy()
        frame_id = h5["frame_id"][index]
        if isinstance(frame_id, (bytes, bytearray)):
            frame_id = frame_id.decode("utf-8")

        if self.transform is not None:
            clear, foggy, P2, labels, depth = self.transform(clear, foggy, P2, labels, depth)

        bbox3d_state = np.zeros([len(labels), 7], dtype=np.float32)
        if len(labels) > 0:
            labels, bbox3d_state = self._reproject(P2, labels)

        bbox2d = (
            np.array([[o.bbox_l, o.bbox_t, o.bbox_r, o.bbox_b] for o in labels], dtype=np.float32)
            if labels
            else np.zeros((0, 4), dtype=np.float32)
        )

        return {
            "calib": P2,
            "image": clear,
            "foggy": foggy,
            "label": [o.type for o in labels],
            "bbox2d": bbox2d,  # [N, 4] x1, y1, x2, y2
            "bbox3d": bbox3d_state,  # [N, 7] proj_cx, proj_cy, z, w, h, l, alpha
            "original_shape": original_shape,
            "depth": depth,
            "original_P": original_P,
            "frame_id": frame_id,
        }

    # ------------------------------------------------------------------ batching
    # Collation lives in ``collate.py`` so DataLoader wiring can import it without
    # the dataset class. Exposed here as a staticmethod for backward compatibility.
    collate_fn = staticmethod(collate_fn)
