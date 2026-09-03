"""Tier 1: single-camera detection and 2D tracking."""
from .pipeline import iter_frames, run, load_config, build_pipeline, FrameOutput

__all__ = ["iter_frames", "run", "load_config", "build_pipeline", "FrameOutput"]
