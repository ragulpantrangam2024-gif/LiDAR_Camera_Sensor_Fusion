
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
STAGE_DIR = Path(__file__).resolve().parent

FRAME_ID = "0000000100"

LIDAR_PATH = (
    PROJECT_DIR
    / "data_3d_raw"
    / "2013_05_28_drive_0000_sync"
    / "velodyne_points"
    / "data"
    / f"{FRAME_ID}.bin"
)

RESULTS_DIR = STAGE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not LIDAR_PATH.exists():
        raise FileNotFoundError(
            f"LiDAR file not found: {LIDAR_PATH}"
        )

    # Each point contains x, y, z and reflectance.
    raw_points = np.fromfile(LIDAR_PATH, dtype=np.float32)

    if raw_points.size % 4 != 0:
        raise ValueError(
            f"Expected four float values per point; "
            f"received {raw_points.size} values."
        )

    points = raw_points.reshape(-1, 4)

    # Remove points with invalid coordinates or reflectance.
    points = points[np.isfinite(points).all(axis=1)]

    x = points[:, 0]
    y = points[:, 1]
    reflectance = points[:, 3]

    print("=" * 60)
    print("STAGE 1 — TASK 5: VISUALIZE LIDAR POINT CLOUD")
    print("=" * 60)
    print(f"Frame: {FRAME_ID}")
    print(f"Valid points: {len(points):,}")
    print(f"X range: {x.min():.2f} to {x.max():.2f} m")
    print(f"Y range: {y.min():.2f} to {y.max():.2f} m")
    print(
        f"Reflectance range: "
        f"{reflectance.min():.3f} to {reflectance.max():.3f}"
    )

    output_path = RESULTS_DIR / f"{FRAME_ID}_lidar_topdown.png"

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        x,
        y,
        c=reflectance,
        s=0.5,
        cmap="viridis",
        linewidths=0,
    )

    ax.set_title(f"LiDAR Top-Down View — Frame {FRAME_ID}")
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig.colorbar(scatter, ax=ax, label="Reflectance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")
    print("\nTask 5 completed successfully.")


if __name__ == "__main__":
    main()
