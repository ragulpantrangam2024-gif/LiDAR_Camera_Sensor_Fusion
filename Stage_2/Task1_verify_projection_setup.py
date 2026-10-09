
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

FRAME_ID = "0000000100"

IMAGE_PATH = PROJECT_DIR / "Stage_1" / "Images" / f"{FRAME_ID}.png"
LIDAR_PATH = (
    PROJECT_DIR / "data_3d_raw" / "2013_05_28_drive_0000_sync"
    / "velodyne_points" / "data" / f"{FRAME_ID}.bin"
)
CALIB_DIR = PROJECT_DIR / "calibration"


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

    height, width = image.shape[:2]

    raw = np.fromfile(LIDAR_PATH, dtype=np.float32)

    if raw.size % 4:
        raise ValueError("LiDAR file does not contain four values per point.")

    points_velo = raw.reshape(-1, 4)[:, :3]
    points_velo = points_velo[np.isfinite(points_velo).all(axis=1)]

    # This file contains a 3x4 camera-to-Velodyne transform.
    transform_values = np.fromfile(
        CALIB_DIR / "calib_cam_to_velo.txt",
        dtype=np.float64,
        sep=" ",
    )

    if transform_values.size != 12:
        raise ValueError("Expected 12 values in calib_cam_to_velo.txt.")

    T_cam_to_velo = np.eye(4)
    T_cam_to_velo[:3, :] = transform_values.reshape(3, 4)

    # Invert to transform Velodyne points into camera coordinates.
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
    rect_size = calibration["S_rect_00"].astype(int)

    points_rect = (R_rect @ points_cam.T).T

    # Only points in front of the camera can be projected.
    in_front = points_rect[:, 2] > 0
    points_front = points_rect[in_front]

    points_front_h = np.column_stack(
        [points_front, np.ones(len(points_front))]
    )
    projected = (P_rect @ points_front_h.T).T

    u = projected[:, 0] / projected[:, 2]
    v = projected[:, 1] / projected[:, 2]

    rect_width, rect_height = rect_size

    inside_rect = (
        (u >= 0) & (u < rect_width)
        & (v >= 0) & (v < rect_height)
    )

    print("=" * 60)
    print("STAGE 2 — TASK 1: VERIFY PROJECTION SETUP")
    print("=" * 60)
    print(f"Frame: {FRAME_ID}")
    print(f"Input image dimensions: {width} x {height}")
    print(
        f"Rectified calibration dimensions: "
        f"{rect_width} x {rect_height}"
    )
    print(f"Valid LiDAR points: {len(points_velo):,}")
    print(f"Points in front of camera: {len(points_front):,}")
    print(f"Projected points inside rectified bounds: {inside_rect.sum():,}")

    print("\nFirst five projected coordinates (u, v):")
    for px, py in zip(u[:5], v[:5]):
        print(f"({px:.2f}, {py:.2f})")

    if (width, height) != (rect_width, rect_height):
        print(
            "\nNOTE: Input image dimensions differ from the rectified "
            "calibration dimensions. Do not overlay yet; verify whether "
            "the input image is raw or rectified."
        )

    print("\nProjection setup check completed.")


if __name__ == "__main__":
    main()
