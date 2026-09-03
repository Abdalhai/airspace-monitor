"""
live.py -- runtime-configurable live camera path. METHOD DEFINITIONS ONLY.

Nothing here is implemented, and that is deliberate. This file exists so the
shape of the live extension is visible in the codebase rather than described in
a document: every method below raises NotImplementedError with a note on what
implementing it actually requires.

The point being made by this file: the live path is not a different system. It
is the same three interfaces from scene.py with different sources behind them.
Dataset and live differ only in where (pose, detection, timestamp) triples come
from; the voxel accumulation, clustering and tracking downstream are unchanged.

WHAT IS ALREADY DONE
    Phase 1's capture.py already handles live ingest -- webcam, RTSP, and the
    Android "IP Webcam" MJPEG endpoint -- including the distinction between a
    dropped frame (retry) and end-of-stream (terminate), and reconnection.
    That part is not stubbed; it works.

WHAT IS ACTUALLY MISSING
    Only the pose problem. Camera POSITIONS are easy to obtain live (tape
    measure, or a phone GNSS fix). Camera ORIENTATIONS are the hard part, and
    they are what dominates 3D error -- measured at ~7 deg on dataset 3, which
    costs 4-7 m of position accuracy. Getting orientation below 1 deg live
    requires a deliberate calibration step, which is what calibrate_*() below
    would provide.

CONFIGURATION AT RUNTIME
    Following the Phase 1 convention, every parameter here is intended to come
    from a YAML block rather than being hardcoded, so cameras can be added or
    moved without touching code:

        live:
          window_seconds: 1.0
          grid:
            voxel_size: 1.0
            center: [0, 0, 50]
            span: 200
          cameras:
            - id: 0
              source: {type: ip_http, url: "http://192.168.1.23:8080/video"}
              position: [0.0, 0.0, 1.5]
              calibration: "calib/phone_a.json"
              orientation: {method: landmarks, landmarks_file: "survey/site_a.txt"}
            - id: 1
              source: {type: rtsp, url: "rtsp://..."}
              position: [40.0, 0.0, 1.5]
              calibration: "calib/gopro.json"
              orientation: {method: checkerboard, images: "calib/cam1/*.jpg"}
"""

from typing import List, Optional, Dict
import numpy as np

from .scene import Scene, Camera, Ray


class LiveScene(Scene):
    """A Scene backed by live camera feeds instead of dataset files.

    Consumes the same interface as Dataset3Scene / Dataset5Scene, so
    fod.voxel and fod.track3d work against it without modification.
    """

    def __init__(self, config: dict):
        self.config = config
        self.cameras: List[Camera] = []
        self._sources: Dict[int, object] = {}
        self._buffer: Dict[int, list] = {}
        raise NotImplementedError(
            "LiveScene is a stub. Implement open_sources() and resolve_poses() "
            "first; both are documented below."
        )

    # -- ingest -------------------------------------------------------------

    def open_sources(self):
        """Instantiate one VideoSource per configured camera.

        Phase 1's capture.create_video_source() already does this and already
        supports file / webcam / rtsp / ip_http. This method is a loop over the
        config's `cameras` list calling it once per entry with a distinct
        source_id. capture.VideoSource was written not to assume a single
        source, so no change is needed there.
        """
        raise NotImplementedError

    def poll(self) -> Dict[int, "FrameResult"]:
        """Read one frame from each source, non-blocking.

        Must handle the FrameStatus cases capture.py already distinguishes:
        OK, DROPPED_OR_LATE (retry, do not terminate), END_OF_FILE,
        SOURCE_LOST (drop that camera, continue with the rest -- a monitoring
        system should degrade to fewer cameras, not stop).
        """
        raise NotImplementedError

    # -- timing -------------------------------------------------------------

    def resolve_timestamps(self, frames) -> Dict[int, float]:
        """Map per-camera arrival times onto one shared clock.

        THIS IS THE HARD PART OF GOING LIVE and it has no dataset equivalent:
        datasets 3, 4 and 5 all ship ground-truth sync, so nothing upstream of
        here has ever had to solve it.

        Wall-clock arrival time is not good enough on its own -- network jitter
        on an MJPEG stream is tens of milliseconds, and a drone at 15 m/s moves
        0.15 m in 10 ms. Options, roughly in increasing order of effort:
          - NTP-discipline every capture device, use arrival time
          - cross-correlate an audio transient (dataset 5 used an audio trigger
            box, see assets/sync_overview.jpg)
          - flash a light in all fields of view and align the frame it appears

        Until this is solved, window_seconds must stay large enough that sync
        error is small relative to the window.
        """
        raise NotImplementedError

    # -- pose ---------------------------------------------------------------

    def resolve_poses(self):
        """Populate self.cameras with resolved centre + R + K + dist.

        Positions come from config. Orientations come from calibrate_* below.
        """
        raise NotImplementedError

    def calibrate_orientation_from_landmarks(self, cam_id: int,
                                            landmarks_world: np.ndarray,
                                            landmarks_pixel: np.ndarray) -> np.ndarray:
        """Solve one camera's rotation by sighting surveyed landmarks.

        Same structure as the dataset 5 approach: with the camera centre known
        and fixed, only 3 rotational DOF remain, so cv2.solvePnP with the centre
        constrained is sufficient. Needs >= 3 non-collinear landmarks with known
        world coordinates; 5-6 spread across the field of view is comfortable.

        Expected accuracy < 1 deg, versus ~7 deg from the dataset 3
        moving-object bundle adjustment -- which is the difference between
        4-7 m and sub-metre 3D error.
        """
        raise NotImplementedError

    def calibrate_orientation_from_checkerboard(self, cam_id: int,
                                                images, board_spec) -> np.ndarray:
        """Alternative: recover rotation from a checkerboard at a known pose.

        More accurate than landmarks and needs no survey equipment, but the
        board must be placed at a known position and orientation in the world
        frame, which in practice is harder outdoors than sighting landmarks.
        """
        raise NotImplementedError

    # -- Scene interface ----------------------------------------------------

    def time_range(self):
        raise NotImplementedError("live streams are unbounded; use poll() instead")

    def time_unit(self):
        return "wall-clock seconds"

    def seconds_per_time_unit(self):
        return 1.0

    def _detection_at(self, cam_id: int, t: float) -> Optional[np.ndarray]:
        """Detections must come from the Phase 1 MOG2 pipeline run per camera,
        buffered into self._buffer, then queried by timestamp.

        NOTE: detections must be in NATIVE pixel coordinates. If Phase 1's
        standardizer resized the frame for detection, scale back up before
        returning, or the K matrix in Camera is invalid. This is the single
        easiest way to silently corrupt the geometry.
        """
        raise NotImplementedError

    def rays_in_window(self, t0: float, t1: float, n_samples: int) -> List[Ray]:
        """Live version buffers rays as frames arrive rather than sampling a
        stored track, but returns the same List[Ray] the accumulator expects,
        so fod.voxel.grid.accumulate_windows() needs no live-specific branch.
        """
        raise NotImplementedError


class StreamingWindowRunner:
    """Drives LiveScene continuously: buffer a window, accumulate, cluster,
    track, emit, repeat.

    Deliberately not implemented. The batch equivalent
    (fod.voxel.grid.accumulate_windows) already expresses the same loop, so
    this is a scheduling and buffering problem rather than an algorithmic one.

    Latency floor is one window -- an answer for 'now' cannot exist until the
    window closes. That is inherent to voxel accumulation and is why this
    system is framed as monitoring with a real-time detection tier, rather
    than as real-time tracking.
    """

    def __init__(self, scene: LiveScene, spec, window_seconds: float = 1.0):
        raise NotImplementedError

    def run_forever(self):
        raise NotImplementedError
