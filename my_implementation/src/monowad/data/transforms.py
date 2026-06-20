"""Runtime augmentation pipeline for the HDF5-backed KittiMonoDataset.

Ported from ``visualDet3D/data/pipeline/stereo_augmentator.py``, but pared down to what
is *not* already baked into ``data.h5``. ``CropTop`` and ``Resize`` are applied at pack
time (the stored images are a fixed ``288x1280`` with the matching post-resize ``P2``),
so the only transforms left to run per ``__getitem__`` are:

* **train** : ConvertToFloat -> PhotometricDistort -> RandomMirror -> Normalize
* **val/test** : ConvertToFloat -> Normalize

Each transform is a callable with the dataset's injected signature::

    (clear, foggy, P2, labels, depth) -> (clear, foggy, P2, labels, depth)

``clear``/``foggy`` are ``[H, W, 3]`` arrays (uint8 on the way in, float32 after
``ConvertToFloat``), ``P2`` is the ``[3, 4]`` post-resize calibration, ``labels`` is a
list of ``SimpleNamespace`` KITTI objects (attribute access: ``bbox_l/t/r/b``,
``x/y/z/w/h/l/ry/alpha``), and ``depth`` is the ¼-res ``[72, 320]`` GT (``None`` on val/test).

Two deliberate departures from the legacy code (see ``CLAUDE.md`` "Key design decisions"):

1. **No clear/foggy (and no P2/P3) role swap on mirror.** The legacy ``RandomMirror``
   swaps ``left_image``/``foggy`` and ``P2``/``P3`` because in a stereo rig a horizontal
   flip swaps the left/right cameras. MonoWAD reuses the right slot for the *foggy* image,
   where that swap is geometrically meaningless and breaks the clean-reference asymmetry.
   We flip clear/foggy/depth/P2/labels **in place** without swapping roles.
2. **PhotometricDistort shares its random parameters across clear & foggy.** The colour
   jitter is sampled once and applied to both images so the ``foggy - clear`` weather
   residual stays coherent (the legacy primitives already behave this way; preserved here).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy import random

from ..utils.geometry import theta2alpha_3d

Sample = Tuple[
    np.ndarray, Optional[np.ndarray], np.ndarray, List, Optional[np.ndarray]
]


class Compose:
    """Chain transforms, threading the full ``(clear, foggy, P2, labels, depth)`` tuple."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        for t in self.transforms:
            clear, foggy, P2, labels, depth = t(clear, foggy, P2, labels, depth)
        return clear, foggy, P2, labels, depth


class ConvertToFloat:
    """Cast the image(s) to ``float32`` (calibration / labels / depth untouched)."""

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        clear = clear.astype(np.float32)
        if foggy is not None:
            foggy = foggy.astype(np.float32)
        return clear, foggy, P2, labels, depth


class Normalize:
    """ImageNet-style normalisation: ``/255``, subtract mean, divide by std (per channel).

    Applied to ``clear`` and ``foggy`` only — depth GT is left in metric units.
    """

    def __init__(self, mean, std) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def _norm(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        reps = int(img.shape[2] / self.mean.shape[0])
        img -= np.tile(self.mean, reps)
        img /= np.tile(self.std, reps)
        return img.astype(np.float32)

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        clear = self._norm(clear)
        if foggy is not None:
            foggy = self._norm(foggy)
        return clear, foggy, P2, labels, depth


# --------------------------------------------------------------------------- photometric
# All photometric primitives sample their random parameter once and apply the *same*
# value to clear and foggy, keeping the weather residual coherent. They assume in-place
# mutation of float32 arrays (PhotometricDistort hands them copies).

class RandomBrightness:
    def __init__(self, distort_prob, delta=32) -> None:
        assert 0.0 <= delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if random.rand() <= self.distort_prob:
            delta = random.uniform(-self.delta, self.delta)
            clear += delta
            if foggy is not None:
                foggy += delta
        return clear, foggy, P2, labels, depth


class RandomContrast:
    def __init__(self, distort_prob, lower=0.5, upper=1.5) -> None:
        assert upper >= lower >= 0, "contrast bounds must satisfy upper >= lower >= 0."
        self.lower, self.upper, self.distort_prob = lower, upper, distort_prob

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if random.rand() <= self.distort_prob:
            alpha = random.uniform(self.lower, self.upper)
            clear *= alpha
            if foggy is not None:
                foggy *= alpha
        return clear, foggy, P2, labels, depth


class ConvertColor:
    """Convert between RGB and HSV colour spaces (deterministic)."""

    def __init__(self, current="RGB", transform="HSV") -> None:
        self.current = current
        self.transform = transform

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if self.current == "RGB" and self.transform == "HSV":
            code = cv2.COLOR_RGB2HSV
        elif self.current == "HSV" and self.transform == "RGB":
            code = cv2.COLOR_HSV2RGB
        else:
            raise NotImplementedError(f"{self.current}->{self.transform}")
        clear = cv2.cvtColor(clear, code)
        if foggy is not None:
            foggy = cv2.cvtColor(foggy, code)
        return clear, foggy, P2, labels, depth


class RandomSaturation:
    """Scale the S channel — assumes the image is already in HSV."""

    def __init__(self, distort_prob, lower=0.5, upper=1.5) -> None:
        assert upper >= lower >= 0, "saturation bounds must satisfy upper >= lower >= 0."
        self.lower, self.upper, self.distort_prob = lower, upper, distort_prob

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if random.rand() <= self.distort_prob:
            ratio = random.uniform(self.lower, self.upper)
            clear[:, :, 1] *= ratio
            if foggy is not None:
                foggy[:, :, 1] *= ratio
        return clear, foggy, P2, labels, depth


class RandomHue:
    """Shift the H channel (wrapping at 360) — assumes the image is already in HSV."""

    def __init__(self, distort_prob, delta=18.0) -> None:
        assert 0.0 <= delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    @staticmethod
    def _shift(img: np.ndarray, shift: float) -> None:
        img[:, :, 0] += shift
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if random.rand() <= self.distort_prob:
            shift = random.uniform(-self.delta, self.delta)
            self._shift(clear, shift)
            if foggy is not None:
                self._shift(foggy, shift)
        return clear, foggy, P2, labels, depth


class PhotometricDistort:
    """Brightness + (contrast, saturation, hue) jitter, with contrast first or last.

    Operates on float32 copies (the originals are untouched). Contrast is duplicated so it
    can run before *or* after the HSV jitter with equal probability; the chosen ordering and
    every sampled parameter are shared between clear and foggy.
    """

    def __init__(
        self,
        distort_prob=1.0,
        contrast_lower=0.5,
        contrast_upper=1.5,
        saturation_lower=0.5,
        saturation_upper=1.5,
        hue_delta=18.0,
        brightness_delta=32,
    ) -> None:
        self.transforms = [
            RandomContrast(distort_prob, contrast_lower, contrast_upper),
            ConvertColor(current="RGB", transform="HSV"),
            RandomSaturation(distort_prob, saturation_lower, saturation_upper),
            RandomHue(distort_prob, hue_delta),
            ConvertColor(current="HSV", transform="RGB"),
            RandomContrast(distort_prob, contrast_lower, contrast_upper),
        ]
        self.rand_brightness = RandomBrightness(distort_prob, brightness_delta)

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        # contrast first (drop the trailing one) or last (drop the leading one).
        distortion = self.transforms[:-1] if random.rand() <= 0.5 else self.transforms[1:]
        distortion = Compose([self.rand_brightness, *distortion])
        clear = clear.copy()
        foggy = None if foggy is None else foggy.copy()
        return distortion(clear, foggy, P2, labels, depth)


# --------------------------------------------------------------------------------- mirror
class RandomMirror:
    """Horizontal flip of clear/foggy/depth/P2/labels, in lock-step and in place.

    Unlike the legacy stereo transform, neither the clear/foggy images nor P2/P3 swap
    roles (MonoWAD's foggy image is not a right-camera view). Label X / yaw / alpha and
    the 2D box are flipped through the (mirrored) P2 exactly as in the legacy code.
    """

    def __init__(self, mirror_prob) -> None:
        self.mirror_prob = mirror_prob

    def __call__(self, clear, foggy=None, P2=None, labels=None, depth=None) -> Sample:
        if random.rand() > self.mirror_prob:
            return clear, foggy, P2, labels, depth

        width = clear.shape[1]
        clear = np.ascontiguousarray(clear[:, ::-1, :])
        if foggy is not None:
            foggy = np.ascontiguousarray(foggy[:, ::-1, :])
        if depth is not None:
            depth = np.ascontiguousarray(depth[:, ::-1])

        if P2 is not None:
            P2[0, 3] = -P2[0, 3]
            P2[0, 2] = width - P2[0, 2] - 1

        if labels:
            for obj in labels:
                # 2D box: reflect across the vertical axis (the tight 3D-projected box is
                # recomputed in the dataset's _reproject afterwards).
                obj.bbox_l, obj.bbox_r = width - obj.bbox_r - 1, width - obj.bbox_l - 1
                # 3D centre: mirror X about the camera axis.
                obj.x = -obj.x
                # Global yaw under the flip.
                ry = obj.ry
                ry = (-math.pi - ry) if ry < 0 else (math.pi - ry)
                while ry > math.pi:
                    ry -= 2 * math.pi
                while ry < -math.pi:
                    ry += 2 * math.pi
                obj.ry = ry
                # Observation angle through the mirrored P2.
                obj.alpha = theta2alpha_3d(ry, obj.x, obj.z, P2)

        return clear, foggy, P2, labels, depth


# --------------------------------------------------------------------------- factories
def build_train_transform(mean, std, mirror_prob: float = 0.5, distort_prob: float = 1.0) -> Compose:
    """Training pipeline: ConvertToFloat -> PhotometricDistort -> RandomMirror -> Normalize."""
    return Compose([
        ConvertToFloat(),
        PhotometricDistort(distort_prob=distort_prob),
        RandomMirror(mirror_prob),
        Normalize(mean, std),
    ])


def build_eval_transform(mean, std) -> Compose:
    """Val / test pipeline: ConvertToFloat -> Normalize (deterministic)."""
    return Compose([
        ConvertToFloat(),
        Normalize(mean, std),
    ])
