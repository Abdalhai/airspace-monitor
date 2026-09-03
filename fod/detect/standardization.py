"""
standardization.py — normalize frames from heterogeneous sources to a common baseline.

Two independent knobs, both configurable in config.yaml:

  1. Resolution: plain cv2.resize to a target size. Cheap, always applied.

  2. Brightness/contrast normalization — this is the "meaningfully better vs.
     simpler tradeoff" the prompt asked to flag rather than silently resolve:

       - "hist_eq" (cv2.equalizeHist, global histogram equalization)
             Cheapest option. Works on one global histogram for the whole frame.
             For sky/cloud backgrounds — large, mostly-uniform bright regions,
             which is exactly this project's domain — global equalization tends
             to over-amplify noise in the sky and can shift contrast unpredictably
             frame to frame. That instability feeds directly into background
             subtraction quality (MOG2 is sensitive to frame-to-frame lighting jitter).

       - "clahe" (Contrast Limited Adaptive Histogram Equalization)
             Slightly more compute than hist_eq, but still cheap and real-time on
             CPU at the resolutions this project targets (≤ ~720p). Operates on
             local tiles with a clip limit, so it does not blow out large uniform
             regions the way global equalization does. More stable frame-to-frame.

       - "none"
             Skip normalization entirely — useful as a baseline for A/B comparison.

     DEFAULT: "clahe", because sky-background stability matters more here than the
     small extra cost. This is a config setting (standardization.brightness_method),
     not a hardcoded choice — change it in config.yaml if you want to compare.

Kept as its own module with no dependency on capture/tracking so it can be unit
tested by just feeding it frames and inspecting the output.

FIX (defect 6) — ASPECT RATIO AND THE CALIBRATION MATRIX
--------------------------------------------------------
The original resized unconditionally to a fixed 960x540. That is safe for display
but destroys geometry two ways:

  1. Sources whose aspect ratio is not 16:9 get anamorphically squeezed. Dataset 3's
     cameras do not share an aspect ratio, so a fixed target silently applies a
     different horizontal and vertical scale to different cameras.

  2. Any intrinsic matrix K measured at native resolution no longer describes the
     resized frame. A centroid taken from a 960x540 frame and fed to a native-
     resolution K produces a bearing that is wrong by the scale ratio -- and looks
     entirely plausible, which is what makes it dangerous.

The fix has two parts. `preserve_aspect` (default True) fits within the target box
using a single uniform scale, so the anamorphic case cannot arise. And the scale
factors are recorded so `to_native()` can map any pixel coordinate measured on a
standardized frame back to native coordinates.

RULE: standardization is a SPEED optimisation for detection. Coordinates that reach
any geometry stage -- angular velocity, ray generation, PnP -- must be in native
pixels. Always route them through to_native() first.
"""

import cv2
import numpy as np


class FrameStandardizer:
    def __init__(self, target_width: int, target_height: int,
                 brightness_method: str = "clahe",
                 clahe_clip_limit: float = 2.0, clahe_tile_grid: int = 8,
                 preserve_aspect: bool = True):
        self.target_size = (target_width, target_height)
        self.brightness_method = brightness_method
        self.preserve_aspect = preserve_aspect
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                       tileGridSize=(clahe_tile_grid, clahe_tile_grid))
        # scale actually applied by the most recent process() call
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.native_size = None      # (w, h) of the last input frame

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.native_size = (w, h)
        tw, th = self.target_size

        if self.preserve_aspect:
            # One uniform scale for both axes: an anamorphic squeeze cannot occur,
            # and to_native() is a single multiplication rather than two.
            s = min(tw / w, th / h)
            if s >= 1.0:
                # Never upscale. Enlarging adds no detail, costs time, and inflates
                # blob areas so min_blob_area no longer means what it was tuned to.
                s = 1.0
            new_w, new_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
            self.scale_x = self.scale_y = new_w / w
        else:
            new_w, new_h = tw, th
            self.scale_x, self.scale_y = tw / w, th / h

        if (new_w, new_h) != (w, h):
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame = self._normalize_brightness(frame)
        return frame

    def to_native(self, x: float, y: float) -> tuple:
        """Map a coordinate measured on a standardized frame back to native pixels.

        Every centroid that reaches a geometry stage must pass through here. A
        standardized coordinate against a native-resolution K yields a wrong
        bearing with no visible symptom."""
        return (x / self.scale_x, y / self.scale_y)

    def native_scale(self) -> tuple:
        return (self.scale_x, self.scale_y)

    def _normalize_brightness(self, frame: np.ndarray) -> np.ndarray:
        if self.brightness_method == "none":
            return frame

        # Operate in YCrCb so we normalize luminance only and leave color (Cr/Cb) alone —
        # normalizing brightness on raw BGR channels independently distorts color balance.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        if self.brightness_method == "clahe":
            y = self._clahe.apply(y)
        elif self.brightness_method == "hist_eq":
            y = cv2.equalizeHist(y)
        else:
            raise ValueError(f"unknown brightness_method: {self.brightness_method!r}")

        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
