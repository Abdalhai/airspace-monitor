"""
motion_detection.py — background subtraction producing a foreground motion mask.

CPU performance notes (flagged per instructions rather than assumed):

  - MOG2 (used here) is the right default: adaptive Gaussian-mixture background
    model, reasonably fast, stock in OpenCV, good general-purpose behavior.

  - detectShadows=True (OpenCV's own default) roughly doubles per-pixel cost,
    because it additionally classifies shadow pixels. We don't care about shadows
    for small fast-moving airborne objects, so this is disabled by default
    (background_subtraction.detect_shadows: false in config).

  - cv2.createBackgroundSubtractorKNN is generally *heavier* than MOG2 for
    comparable quality — avoid it on CPU-only unless you have a specific reason.

  - If MOG2 at full standardized resolution is too slow for your target FPS on
    the deployment CPU, the cheapest lever (try this before switching algorithms)
    is downscaling the frame before background subtraction — e.g. run bg-sub at
    640x360 even though the display/standardized frame is 960x540 — then scale
    the resulting mask back up. Exposed as `bg_sub_scale` in config (1.0 = off).

  - Morphological open+close on the raw mask is cheap and removes a lot of
    single-pixel sensor noise that would otherwise pollute contour/blob detection
    downstream — included by default, kernel size configurable.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, history: int = 500, var_threshold: float = 16.0,
                 detect_shadows: bool = False, bg_sub_scale: float = 1.0,
                 morph_kernel_size: int = 3):
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.bg_sub_scale = bg_sub_scale
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Returns a binary (0/255) foreground mask at the input frame's resolution."""
        h, w = frame.shape[:2]
        proc_frame = frame
        if self.bg_sub_scale != 1.0:
            proc_frame = cv2.resize(
                frame, (int(w * self.bg_sub_scale), int(h * self.bg_sub_scale)),
                interpolation=cv2.INTER_AREA
            )

        mask = self._sub.apply(proc_frame)
        # If shadows were ever enabled, MOG2 marks them gray (127) — threshold them out.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        if self.bg_sub_scale != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask
