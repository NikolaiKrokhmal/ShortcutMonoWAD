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

* ``validation_step`` / ``on_validation_epoch_*`` — the body of legacy ``test_one`` +
  ``evaluate_kitti_obj``: per frame, run the detector's test path, back-project the 3D
  state to camera coords, remap the 2D box to *original* KITTI resolution via
  ``original_P`` vs the post-resize ``P2``, and write a KITTI result ``.txt``. At epoch
  end the numba KITTI evaluator scores AP and the headline numbers are logged. Controlled
  by ``cfg.eval`` (set ``eval.enabled=false`` for an inference-only sanity pass).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .utils.annotations import compound_annotation
from .utils.geometry import BackProjection, BBox3dProjector
from .eval.result_writer import write_result_to_file


class MonoWADModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.detector = hydra.utils.instantiate(cfg.detector)
        # ordered class names, used to index labels when compounding annotations
        self.obj_types = list(self.detector.obj_types)
        self._warned_no_ap = False

        # eval-time geometry (back-project 3D state to camera coords + recover yaw).
        # Registered as submodules so Lightning moves their buffers to the right device.
        self.backprojector = BackProjection()
        self.projector = BBox3dProjector()

        # KITTI AP eval config (see configs/config.yaml ``eval``); absent -> disabled.
        eval_cfg = cfg.get("eval", None)
        self._eval_enabled = bool(eval_cfg.enabled) if eval_cfg is not None else False
        if self._eval_enabled:
            self._label_dir = eval_cfg.label_dir
            self._result_dir = eval_cfg.result_dir
            self._score_thr = float(eval_cfg.get("score_thr", 0.4))
            self._eval_gpu = int(eval_cfg.get("gpu", 0))
        self._val_frame_ids: list[str] = []

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
    def on_validation_epoch_start(self) -> None:
        # Fresh result dir each epoch so stale .txt files can't leak into scoring.
        if not self._eval_enabled:
            return
        self._val_frame_ids = []
        if os.path.isdir(self._result_dir):
            shutil.rmtree(self._result_dir)
        os.makedirs(self._result_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        # val collate -> (rgb, calib, labels, bbox2d, bbox3d, foggy, original_Ps, frame_ids)
        rgb, calib = batch[0], batch[1]
        foggy = batch[5] if len(batch) > 5 else None
        original_Ps = batch[6] if len(batch) > 6 else None
        frame_ids = batch[7] if len(batch) > 7 else None

        # test_forward expects batch size 1 (legacy constraint)
        num_det = 0
        for i in range(rgb.shape[0]):
            img_i = rgb[i : i + 1].float().contiguous()
            calib_i = calib[i : i + 1].float()
            foggy_i = foggy[i : i + 1].float().contiguous() if foggy is not None else None
            scores, bboxes, cls_idx = self.detector(
                [img_i, calib_i, foggy_i], eval_weather_type="clear"
            )
            num_det += int(scores.shape[0])

            if self._eval_enabled and frame_ids is not None:
                self._write_kitti_result(
                    scores, bboxes, cls_idx, calib_i[0], original_Ps[i], frame_ids[i]
                )

        self.log("val/num_detections", float(num_det), batch_size=rgb.shape[0])

    @staticmethod
    def _remap_2d(bbox_2d: torch.Tensor, P2: torch.Tensor, original_P: torch.Tensor) -> torch.Tensor:
        """Map 2D boxes from the post-resize (288x1280) frame back to original KITTI
        resolution. Folds CropTop + Resize into one affine derived from the two calib
        matrices (legacy ``test_one``); the crop offset falls out of ``shift_top``."""
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]
        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top
        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y
        return bbox_2d

    def _write_kitti_result(self, scores, bboxes, cls_idx, P2, original_P, frame_id) -> None:
        """Per-frame port of legacy ``test_one``: 3D back-projection + 2D remap, then write
        the KITTI result file named by ``frame_id``."""
        self._val_frame_ids.append(frame_id)
        P2 = P2.detach()  # [3, 4] post-resize calib the model saw

        if bboxes.shape[0] == 0:
            # write an empty file so the frame still participates in (and lowers) recall
            write_result_to_file(self._result_dir, frame_id, [], np.zeros((0, 4), np.float32))
            return

        bbox_2d = bboxes[:, 0:4].clone()
        bbox_3d_state = bboxes[:, 4:]  # [N, 7] proj_cx, proj_cy, z, w, h, l, alpha
        bbox_3d_state_3d = self.backprojector(bbox_3d_state, P2)  # [N, 7] camera coords
        _, _, thetas = self.projector(bbox_3d_state_3d, P2)

        original_P = torch.as_tensor(original_P, dtype=P2.dtype, device=P2.device)
        bbox_2d = self._remap_2d(bbox_2d, P2, original_P)

        obj_names = [self.obj_types[int(c)] for c in cls_idx]
        write_result_to_file(
            self._result_dir,
            frame_id,
            scores.detach().cpu().numpy(),
            bbox_2d.detach().cpu().numpy(),
            bbox_3d_state_3d.detach().cpu().numpy(),
            thetas.detach().cpu().numpy(),
            obj_types=obj_names,
            threshold=self._score_thr,
        )

    def on_validation_epoch_end(self) -> None:
        if not self._eval_enabled:
            if not self._warned_no_ap:
                print("[MonoWADModule] eval.enabled=false; validation ran inference only.")
                self._warned_no_ap = True
            return
        if not self._val_frame_ids:
            return

        # Lazy import: pulls in the GPU-only numba rotated-IoU kernels (see kitti_eval).
        from .eval.kitti_eval import evaluate_kitti, parse_ap

        current_classes = list(range(len(self.obj_types)))
        try:
            result_texts = evaluate_kitti(
                self._result_dir, self._label_dir, self._val_frame_ids,
                current_classes, gpu=self._eval_gpu,
            )
        except Exception as exc:  # don't let a scoring hiccup kill the whole run
            print(f"[MonoWADModule] KITTI AP scoring failed: {exc}")
            return

        for cls_i, text in zip(current_classes, result_texts):
            cls_name = self.obj_types[cls_i]
            print(text)
            ap = parse_ap(text)
            for metric in ("3d", "bev", "bbox"):
                vals = ap.get(metric)
                if vals and len(vals) == 3:
                    easy, mod, hard = vals
                    self.log(f"val/{cls_name}_{metric}_easy", easy)
                    self.log(f"val/{cls_name}_{metric}_mod", mod, prog_bar=(metric == "3d"))
                    self.log(f"val/{cls_name}_{metric}_hard", hard)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
