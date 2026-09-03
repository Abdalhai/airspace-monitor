"""
pipeline.py -- single-camera detection pipeline (Tier 1).

    capture -> standardize -> background subtract -> blobs -> track -> annotate

WHAT CHANGED FROM main.py, AND WHY
----------------------------------
The original was a script: it ran a loop and drew windows. That is fine standalone
but it cannot be composed -- the CLI could not obtain per-frame detections without
either duplicating the loop or importing a module that immediately started
displaying video.

So the loop is now a GENERATOR. It yields one FrameOutput per frame and holds no
opinion about what the caller does with it. Display, video writing, angular
velocity and ray generation all become consumers rather than things the loop must
know about. `run()` reproduces the original script behaviour on top of it, so
`python -m fod.detect.pipeline --config config.yaml` still works exactly as before.

This is what lets the CLI's `angular` command use the real detector instead of
carrying a second copy of it. Two detectors in one submission is the kind of thing
a reviewer notices immediately.

CENTROID COORDINATES
--------------------
Every FrameOutput carries centroids twice: `tracks` in standardized pixels for
drawing, and `centroids_native` in native sensor pixels for geometry. Anything
touching a calibration matrix must use the latter. See standardization.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Iterator, Optional

import cv2
import numpy as np
import yaml

from .capture import create_video_source, FrameStatus
from .standardization import FrameStandardizer
from .motion_detection import MotionDetector
from .tracking import CentroidTracker, extract_blobs
from .visualization import draw_tracks


@dataclass
class FrameOutput:
    """One frame's worth of results."""
    frame_index: int                 # count of frames this run has processed
    native_frame_id: int             # decoder frame number; join key for timestamps
    media_time_s: float              # native_frame_id / native_fps
    wall_time_s: float
    frame: np.ndarray                # standardized frame
    mask: np.ndarray
    tracks: list                     # Track objects, centroids in STANDARDIZED px
    centroids_native: dict = field(default_factory=dict)   # {track_id: (x, y)} NATIVE px
    scale: tuple = (1.0, 1.0)

    @property
    def confirmed_tracks(self) -> list:
        return [t for t in self.tracks if t.confirmed]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict):
    source = create_video_source("cam0", cfg["source"])
    std_cfg = cfg["standardization"]
    std = FrameStandardizer(
        target_width=std_cfg["target_width"],
        target_height=std_cfg["target_height"],
        brightness_method=std_cfg["brightness_method"],
        clahe_clip_limit=std_cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_grid=std_cfg.get("clahe_tile_grid", 8),
        preserve_aspect=std_cfg.get("preserve_aspect", True),
    )
    bg_cfg = cfg["background_subtraction"]
    bgsub = MotionDetector(
        history=bg_cfg["history"],
        var_threshold=bg_cfg["var_threshold"],
        detect_shadows=bg_cfg["detect_shadows"],
        bg_sub_scale=bg_cfg.get("bg_sub_scale", 1.0),
        morph_kernel_size=bg_cfg.get("morph_kernel_size", 3),
    )
    tr_cfg = cfg["tracking"]
    tracker = CentroidTracker(
        max_match_distance=tr_cfg["max_match_distance"],
        max_missed_frames=tr_cfg["max_missed_frames"],
        history_len=tr_cfg["history_len"],
        # defect 1: this was configurable but never passed. Every direction used
        # the hardcoded default regardless of what the config said.
        direction_lookback=tr_cfg.get("direction_lookback", 5),
        min_hits=tr_cfg.get("min_hits", 3),      # defect 4
    )
    return source, std, bgsub, tracker


def iter_frames(cfg: dict, max_frames: Optional[int] = None) -> Iterator[FrameOutput]:
    """Yield one FrameOutput per successfully processed frame.

    The caller owns the loop, so it can stop early, tee results to several
    consumers, or run headless without the generator knowing or caring.
    """
    source, std, bgsub, tracker = build_pipeline(cfg)
    min_area = cfg["background_subtraction"].get("min_blob_area", 25)
    frame_index = 0

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            result = source.read()

            if result.status == FrameStatus.END_OF_FILE:
                break
            if result.status == FrameStatus.SOURCE_LOST:
                break
            if result.status == FrameStatus.DROPPED_OR_LATE:
                # A live hiccup is not end-of-stream. Keep polling.
                continue

            frame = std.process(result.frame)
            mask = bgsub.apply(frame)
            blobs = extract_blobs(mask, min_area=min_area)
            tracks = tracker.update(blobs, frame_index)

            native = {}
            for t in tracks:
                if t.history:
                    native[t.track_id] = std.to_native(*t.history[-1])

            yield FrameOutput(
                frame_index=frame_index,
                native_frame_id=result.native_frame_id,
                media_time_s=result.media_time_s,
                wall_time_s=result.timestamp,
                frame=frame, mask=mask, tracks=tracks,
                centroids_native=native, scale=std.native_scale(),
            )
            frame_index += 1
    finally:
        source.release()


def run(cfg: dict, headless: bool = False, max_frames: Optional[int] = None) -> int:
    """Original script behaviour, built on the generator."""
    writer = None
    out_path = cfg["output"].get("write_video_path")
    show_window = cfg["output"].get("show_window", True) and not headless
    lookback = cfg["tracking"].get("direction_lookback", 5)

    # defect 3: the writer used to hardcode 30.0 fps regardless of the source, so
    # any output from a 25 or 60 fps source played at the wrong speed -- and since
    # it played, nothing looked broken.
    fps = cfg["source"].get("native_fps") or 0.0

    n = 0
    try:
        for out in iter_frames(cfg, max_frames=max_frames):
            annotated = draw_tracks(out.frame, out.tracks, mask=out.mask,
                                    lookback=lookback, only_confirmed=True)

            if writer is None and out_path:
                if not fps or fps <= 0:
                    fps = 30.0
                    print("warning: source fps unknown, writing at 30.0 -- "
                          "playback speed will be wrong if the source is not 30 fps")
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (w, h))
            if writer:
                writer.write(annotated)

            if show_window:
                cv2.imshow("Tier 1 - motion + direction", annotated)
                cv2.imshow("mask", out.mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n = out.frame_index + 1
    finally:
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier 1 single-camera pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", default=None, help="override source.path")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg["source"]["type"] = "file"
        cfg["source"]["path"] = args.source

    n = run(cfg, headless=args.headless, max_frames=args.max_frames)
    print(f"processed {n} frames")
    return 0


# Required on Windows, where child processes re-import this module.
if __name__ == "__main__":
    raise SystemExit(main())
