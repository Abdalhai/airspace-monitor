"""
capture.py — Unified video source abstraction.

Supports:
  - local video files (with optional real-time pacing)
  - live sources: webcam index, RTSP URL, IP-camera HTTP URL
    (e.g. the Android "IP Webcam" app's MJPEG endpoint: http://<ip>:8080/video)

Design goal: the rest of the pipeline calls .read() and gets back a uniform
FrameResult regardless of source type. Adding a second simultaneous source later
(for cross-camera work) just means instantiating a second VideoSource with a
different source_id — this class does not assume there is only ever one, so
nothing here needs to be rewritten when that day comes.
"""

import time
import enum
from dataclasses import dataclass
from typing import Optional, Union
import cv2
import numpy as np


class SourceType(enum.Enum):
    FILE = "file"
    LIVE = "live"


class FrameStatus(enum.Enum):
    OK = "ok"
    END_OF_FILE = "end_of_file"      # file source only: no more frames, expected termination
    DROPPED_OR_LATE = "dropped"      # live source only: a read failed but stream is still alive
    SOURCE_LOST = "source_lost"      # live source: reconnect attempts exhausted, give up


@dataclass
class FrameResult:
    status: FrameStatus
    frame: Optional[np.ndarray]
    timestamp: float          # wall-clock time.time() at the moment of capture
    source_id: str
    frame_index: int          # monotonically increasing count of OK frames from this source
    native_frame_id: int = -1 # decoder's own frame number; -1 if unavailable (live streams)
    media_time_s: float = -1.0  # native_frame_id / native_fps; -1.0 if unavailable

    # FIX (defect 2): frame_index counts frames THIS OBJECT successfully read. It is
    # not the frame's position in the file. The two diverge the moment a read is
    # dropped or the caller skips frames, and the drift is silent.
    #
    # That matters because joining detections to an external timestamp table
    # (dataset 5's cam*_frame_ts.txt) is keyed on the decoder's frame number. A
    # frame_index-based join looks fine and produces a wrong time base, which then
    # propagates into every downstream 3D position with no visible symptom.
    #
    # native_frame_id comes from CAP_PROP_POS_FRAMES and is the value to join on.


class VideoSource:
    def __init__(self, source_id: str, uri: Union[str, int], source_type: SourceType,
                 pace_realtime: bool = False, reconnect_attempts: int = 3,
                 reconnect_delay_s: float = 1.0):
        self.source_id = source_id
        self.uri = uri
        self.source_type = source_type
        self.pace_realtime = pace_realtime
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s

        self._cap = cv2.VideoCapture(uri)
        if not self._cap.isOpened():
            raise RuntimeError(f"[{source_id}] could not open source: {uri}")

        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._frame_interval = (1.0 / self._native_fps) if self._native_fps > 1e-3 else 0.0
        self._last_read_time = None
        self._frame_index = 0
        self._consecutive_failures = 0

    @property
    def native_fps(self) -> float:
        return self._native_fps

    def read(self) -> FrameResult:
        if self.pace_realtime and self.source_type == SourceType.FILE and self._frame_interval > 0:
            self._sleep_for_pacing()

        ok, frame = self._cap.read()
        now = time.time()

        if not ok or frame is None:
            if self.source_type == SourceType.FILE:
                # A file running out of frames is expected termination, not an error.
                return FrameResult(FrameStatus.END_OF_FILE, None, now, self.source_id, self._frame_index)
            return self._handle_live_failure(now)

        self._consecutive_failures = 0
        self._last_read_time = now
        self._frame_index += 1

        # Read AFTER the successful read: CAP_PROP_POS_FRAMES reports the index of
        # the NEXT frame to be decoded, so the frame just returned is one before it.
        native_id = -1
        media_t = -1.0
        try:
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos and pos > 0:
                native_id = int(pos) - 1
                if self._native_fps > 1e-3:
                    media_t = native_id / self._native_fps
        except Exception:
            pass   # live streams often do not support the property at all

        return FrameResult(FrameStatus.OK, frame, now, self.source_id,
                           self._frame_index, native_id, media_t)

    def _handle_live_failure(self, now: float) -> FrameResult:
        # A live stream hiccup (dropped frame, brief network blip) is not the same thing
        # as the file ending — the caller should keep polling, not treat it as termination.
        self._consecutive_failures += 1
        if self._consecutive_failures <= self.reconnect_attempts:
            self._cap.release()
            time.sleep(self.reconnect_delay_s)
            self._cap = cv2.VideoCapture(self.uri)
            return FrameResult(FrameStatus.DROPPED_OR_LATE, None, now, self.source_id, self._frame_index)
        return FrameResult(FrameStatus.SOURCE_LOST, None, now, self.source_id, self._frame_index)

    def _sleep_for_pacing(self):
        if self._last_read_time is None:
            return
        elapsed = time.time() - self._last_read_time
        remaining = self._frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def release(self):
        self._cap.release()


def create_video_source(source_id: str, spec: dict) -> VideoSource:
    """
    Factory. `spec` comes straight from config.yaml's `source:` block, e.g.:
      {"type": "file", "path": "test.mp4", "pace_realtime": true}
      {"type": "webcam", "index": 0}
      {"type": "rtsp", "url": "rtsp://..."}
      {"type": "ip_http", "url": "http://192.168.1.23:8080/video"}   # Android IP Webcam app
    """
    kind = spec.get("type")
    if kind == "file":
        return VideoSource(source_id, spec["path"], SourceType.FILE,
                            pace_realtime=spec.get("pace_realtime", False))
    elif kind == "webcam":
        return VideoSource(source_id, int(spec.get("index", 0)), SourceType.LIVE)
    elif kind in ("rtsp", "ip_http"):
        return VideoSource(source_id, spec["url"], SourceType.LIVE,
                            reconnect_attempts=spec.get("reconnect_attempts", 3),
                            reconnect_delay_s=spec.get("reconnect_delay_s", 1.0))
    else:
        raise ValueError(f"unknown source type: {kind!r}")
