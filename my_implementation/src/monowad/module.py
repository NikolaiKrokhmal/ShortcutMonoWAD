"""MonoWADModule — the LightningModule tying detector + optim + eval together.

Inlines the per-step logic that used to live in
``visualDet3D/networks/pipelines/{trainers,evaluators,testers}.py``. Phase 3 of PLAN.md.

What is ported here
-------------------
* ``training_step`` — the body of legacy ``train_mono_detection``: build the compound
  annotation, run the detector's train path, average + sum the loss terms, log them.
  Gradient clipping is delegated to ``Trainer(gradient_clip_val=...)`` (legacy did it
  by hand via ``clip_grad_norm_``); the optimizer step/zero_grad are Lightning's job.
* ``configure_optimizers`` — Adam + CosineAnnealingLR via Hydra ``instantiate``.
* ``validation_step`` — runs the detector's test path (``test_forward``) so validation
  exercises the full inference graph and surfaces the detection count. It is a runnable
  sanity pass.

Known gap (next tier)
---------------------
Full KITTI AP validation is **not** wired yet. It needs two things that are still
missing: (1) the numba KITTI evaluator (``monowad.eval.kitti_eval`` is a stub), and
(2) the eval-time 2D-box remapping (legacy ``test_one``) which requires ``original_P`` /
``original_shape`` to be carried through the val ``collate_fn`` (currently dropped).
``on_validation_epoch_end`` therefore only logs that AP scoring is pending.
"""
from __future__ import annotations

import numpy as np
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .utils.annotations import compound_annotation


class MonoWADModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.detector = hydra.utils.instantiate(cfg.detector)
        # ordered class names, used to index labels when compounding annotations
        self.obj_types = list(self.detector.obj_types)
        self._warned_no_ap = False

        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path:
            self._load_pretrained(ckpt_path)

    def _load_pretrained(self, ckpt_path: str) -> None:
        """Load a pretrained detector state_dict (keys ``mono_core.*`` / ``bbox_head.*``)
        with ``strict=False``, printing a match report so any key drift is visible.

        Accepts a raw ``state_dict`` or a wrapper dict (``state_dict`` / ``model_state_dict``);
        strips a leading ``detector.`` (e.g. a Lightning checkpoint of this module)."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and not any(torch.is_tensor(v) for v in sd.values()):
            for cand in ("state_dict", "model_state_dict", "model", "net"):
                if isinstance(sd.get(cand), dict):
                    sd = sd[cand]
                    break
        sd = {(k[len("detector."):] if k.startswith("detector.") else k): v for k, v in sd.items()}

        result = self.detector.load_state_dict(sd, strict=False)
        matched = len(sd) - len(result.unexpected_keys)
        print(f"[MonoWADModule] loaded pretrained weights from {ckpt_path}")
        print(f"  matched {matched}/{len(sd)} ckpt tensors | "
              f"missing: {len(result.missing_keys)} | unexpected: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print("  missing (first 10):", result.missing_keys[:10])
        if result.unexpected_keys:
            print("  unexpected (first 10):", result.unexpected_keys[:10])

    # ------------------------------------------------------------------ train
    def training_step(self, batch, batch_idx):
        # train collate -> (rgb, calib, labels, bbox2d, bbox3d, depth, foggy)
        rgb, calib, labels, bbox2ds, bbox3ds, depth, foggy = batch

        max_length = int(np.max([len(label) for label in labels]))
        if max_length == 0:  # nothing to supervise this step
            return None

        annotation = compound_annotation(labels, max_length, bbox2ds, bbox3ds, self.obj_types)

        cls_loss, reg_loss, l_proposed, loss_dict = self.detector(
            [
                rgb.float().contiguous(),
                rgb.new(annotation),  # follows rgb's device/dtype
                calib,
                depth.float().contiguous(),
                foggy.float().contiguous(),
            ]
        )

        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss + l_proposed

        if bool(loss == 0):  # degenerate step (e.g. all anchors ignored) — skip
            return None

        bs = rgb.shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/cls_loss", cls_loss, batch_size=bs)
        self.log("train/reg_loss", reg_loss, batch_size=bs)
        self.log("train/proposed_loss", l_proposed, batch_size=bs)
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.mean()
            self.log(f"train/{name}", value, batch_size=bs)

        return loss

    # ------------------------------------------------------------------ val
    def validation_step(self, batch, batch_idx):
        # val collate -> (rgb, calib, labels, bbox2d, bbox3d, foggy); no depth GT
        rgb, calib, labels = batch[0], batch[1], batch[2]
        foggy = batch[5] if len(batch) > 5 else None

        # test_forward expects batch size 1 (legacy constraint)
        num_det = 0
        for i in range(rgb.shape[0]):
            img_i = rgb[i : i + 1].float().contiguous()
            calib_i = calib[i : i + 1].float()
            foggy_i = foggy[i : i + 1].float().contiguous() if foggy is not None else None
            scores, bboxes, _ = self.detector(
                [img_i, calib_i, foggy_i], eval_weather_type="clear"
            )
            num_det += int(scores.shape[0])

        self.log("val/num_detections", float(num_det), batch_size=rgb.shape[0])

    def on_validation_epoch_end(self) -> None:
        # TODO(next tier): port the numba KITTI evaluator + carry original_P/shape
        # through the val collate, then write KITTI result files and self.log() AP.
        if not self._warned_no_ap:
            print(
                "[MonoWADModule] validation ran inference only; KITTI AP scoring is not "
                "wired yet (needs monowad.eval.kitti_eval + eval-time calib remapping)."
            )
            self._warned_no_ap = True

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
