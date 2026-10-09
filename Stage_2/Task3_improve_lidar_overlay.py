
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAME_ID = "0000000100"
MAX_DISPLAY_POINTS = 8000

IMAGE_PATH = PROJECT_DIR / "Stage_1" / "Images" / f"{FRAME_ID}.png"
LIDAR_PATH = (
    PROJECT_DIR / "data_3d_raw" / "2013_05_28_drive_0000_sync"
    / "velodyne_points" / "data" / f"{FRAME_ID}.bin"
)
CALIB_DIR = PROJECT_DIR / "calibration"
RESULTS_DIR = PROJECT_DIR / "Stage_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_named_calibration(path):
    parameters = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            try:
                values = np.array(
                    [float(v) for v in value.split()],
                    dtype=np.float64,
                )
            except ValueError:
                continue

            if values.size:
                parameters[key.strip()] = values

    return parameters


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    raw = np.fromfile(LIDAR_PATH, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("Expected four float values per LiDAR point.")

    points = raw.reshape(-1, 4)
    points = points[np.isfinite(points).all(axis=1)]
    points_velo = points[:, :3]

    transform_values = np.fromfile(
        CALIB_DIR / "calib_cam_to_velo.txt",
        dtype=np.float64,
        sep=" ",
    )

    if transform_values.size != 12:
        raise ValueError("Expected 12 transformation values.")

    # Camera-to-Velodyne transform; invert to get Velodyne-to-camera.
    T_cam_to_velo = np.eye(4)
    T_cam_to_velo[:3, :] = transform_values.reshape(3, 4)
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    points_h = np.column_stack(
        [points_velo, np.ones(len(points_velo))]
    )
    points_cam = (T_velo_to_cam @ points_h.T).T[:, :3]

    calibration = load_named_calibration(
        CALIB_DIR / "perspective.txt"
    )
    R_rect = calibration["R_rect_00"].reshape(3, 3)
    P_rect = calibration["P_rect_00"].reshape(3, 4)

    points_rect = (R_rect @ points_cam.T).T

    # Keep points in front of the camera.
    front_mask = points_rect[:, 2] > 0
    points_front = points_rect[front_mask]

    points_front_h = np.column_stack(
        [points_front, np.ones(len(points_front))]
    )
    projected = (P_rect @ points_front_h.T).T

    depth = projected[:, 2]
    u = projected[:, 0] / depth
    v = projected[:, 1] / depth

    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0) & (u < width)
        & (v >= 0) & (v < height)
    )

    u = u[inside]
    v = v[inside]
    depth = depth[inside]

    if len(u) == 0:
        raise RuntimeError("No projected points fall inside the image.")

    # Deterministic sampling for display only.
    # Statistics above still use every valid projected point.
    step = max(1, int(np.ceil(len(u) / MAX_DISPLAY_POINTS)))
    u_display = u[::step]
    v_display = v[::step]
    depth_display = depth[::step]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(image_rgb)

    scatter = ax.scatter(
        u_display,
        v_display,
        c=depth_display,
        cmap="turbo",
        s=2,
        alpha=0.8,
        linewidths=0,
    )

    ax.set_title(f"Sampled LiDAR–Camera Overlay — {FRAME_ID}")
    ax.set_xlabel("Image u (pixels)")
    ax.set_ylabel("Image v (pixels)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    fig.colorbar(scatter, ax=ax, label="Depth (m)")
    fig.tight_layout()

    output_path = RESULTS_DIR / f"{FRAME_ID}_lidar_overlay_clean.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("=" * 60)
    print("STAGE 2 — TASK 3: IMPROVE LIDAR OVERLAY")
    print("=" * 60)
    print(f"Frame: {FRAME_ID}")
    print(f"Valid LiDAR points: {len(points_velo):,}")
    print(f"Points in front of camera: {len(points_front):,}")
    print(f"Points inside image: {len(u):,}")
    print(f"Points displayed: {len(u_display):,}")
    print(f"Depth range: {depth.min():.2f} to {depth.max():.2f} m")
    print(f"Output saved to: {output_path}")
    print("\nTask 3 completed.")


if __name__ == "__main__":
    main()
