"""Preprocessing: build imdb + anchor statistics (port of legacy scripts/imdb_precompute_3d.py).

Produces data/train/{data.h5, anchor_mean_Car.npy, anchor_std_Car.npy}. Phase 1 of PLAN.md.
"""
import pyrootutils

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=False)

# TODO: port preprocessing logic.
