"""
visualization.py — overlay direction arrows/labels on frames.

This is the concrete success criterion for Phase 1: watching the output video
should directly show each tracked blob with an arrow indicating its detected
direction of motion, not just a log line.
"""

import cv2
import numpy as np


def draw_tracks(frame: np.ndarray, tracks: list, mask: np.ndarray = None,
                lookback: int = None, only_confirmed: bool = True) -> np.ndarray:
    """FIX (defect 1): `lookback` is now accepted and forwarded to
    Track.direction(). Previously this called direction() with no argument, so the
    configured tracking.direction_lookback was silently ignored no matter what the
    config said.

    `only_confirmed` (defect 4) hides tracks that have not yet reached min_hits, so
    single-frame noise blobs do not flash ids across the display."""
    out = frame.copy()

    if mask is not None:
        overlay = out.copy()
        overlay[mask > 0] = (0, 255, 0)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    for track in tracks:
        if not track.history:
            continue
        if only_confirmed and not getattr(track, "confirmed", True):
            continue
        cx, cy = track.history[-1]
        cx, cy = int(cx), int(cy)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        result = track.direction(lookback)
        if result is not None:
            angle, label = result
            length = 35
            ex = int(cx + length * np.cos(np.radians(angle)))
            ey = int(cy - length * np.sin(np.radians(angle)))
            cv2.arrowedLine(out, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.35)
            cv2.putText(out, f"id{track.track_id} {label} ({angle:.0f} deg)",
                        (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, f"id{track.track_id}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return out
