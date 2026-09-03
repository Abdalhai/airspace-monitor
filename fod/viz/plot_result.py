"""plot_result.py -- render the pipeline output as a single demo figure."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def render(track_csv, det_json, poses_json, out_png, title):
    d = np.genfromtxt(track_csv, delimiter=",", names=True)
    t, X, Y, Z, S = d["t"], d["x"], d["y"], d["z"], d["speed_mps"]
    det = json.load(open(det_json))
    raw = np.array([[r["x"], r["y"], r["z"]] for r in det])
    ncam = np.array([r["n_cameras"] for r in det])
    C = np.array([c["center"] for c in json.load(open(poses_json))["cameras"]])

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.28, wspace=0.26)

    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.scatter(raw[:, 0], raw[:, 1], raw[:, 2], s=14, c="#c8c8c8",
               label="raw voxel clusters", depthshade=False)
    ax.plot(X, Y, Z, color="#d1495b", lw=2.4, label="Kalman-filtered 3D track")
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=110, marker="^", c="#0a5c8a",
               edgecolors="k", linewidths=0.6, label="cameras", depthshade=False)
    for i, c in enumerate(C):
        ax.text(c[0], c[1], c[2] + 3, f"cam{i}", fontsize=8, ha="center")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Reconstructed 3D trajectory", fontsize=12, pad=0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.view_init(elev=22, azim=-58)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(X, Y, color="#d1495b", lw=2)
    ax2.scatter(raw[:, 0], raw[:, 1], s=10, c="#c8c8c8", zorder=0)
    ax2.scatter(C[:, 0], C[:, 1], s=70, marker="^", c="#0a5c8a", zorder=3)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("Top-down (X-Y)", fontsize=10)
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, Z, color="#3f7d20", lw=2)
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("altitude Z (m)")
    ax3.set_title("Altitude profile", fontsize=10); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, S, color="#e07a1f", lw=2)
    ax4.fill_between(t, 0, S, color="#e07a1f", alpha=0.18)
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("speed (m/s)")
    ax4.set_title(f"Speed  (mean {S.mean():.1f}, max {S.max():.1f} m/s)", fontsize=10)
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    vals, cnts = np.unique(ncam, return_counts=True)
    ax5.bar(vals, cnts, color="#0a5c8a", alpha=0.85)
    ax5.set_xlabel("cameras supporting cluster"); ax5.set_ylabel("windows")
    ax5.set_title("Ray support per detection", fontsize=10)
    ax5.set_xticks(vals); ax5.grid(alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, y=0.97)
    fig.savefig(out_png, dpi=115, bbox_inches="tight", facecolor="white")
    return dict(n_track=len(t), speed_mean=float(S.mean()), speed_max=float(S.max()),
                duration=float(t[-1] - t[0]))


if __name__ == "__main__":
    r = render("/home/claude/out/track_3d.csv",
               "/home/claude/out/detections_3d.json",
               "/mnt/user-data/uploads/solved_camera_poses.json",
               "/home/claude/out/reconstruction.png",
               "Multi-camera airspace monitoring — dataset 3, 6 cameras, "
               "voxel-grid 3D reconstruction")
    print(r)
