
from pathlib import Path
import csv
import re

import numpy as np
import cv2

# ============================================================
# STAGE 3 - TASK 4: LIDAR-TO-CAR-MASK ASSOCIATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FRAME = "0000000100"

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"

LIDAR_PATH = (
    ROOT / "data_3d_raw"
    / "2013_05_28_drive_0000_sync"
    / "velodyne_points" / "data" / f"{FRAME}.bin"
)

CALIB_DIR = ROOT / "calibration"
CAM_TO_VELO_PATH = CALIB_DIR / "calib_cam_to_velo.txt"
PERSPECTIVE_PATH = CALIB_DIR / "perspective.txt"

MASKS_DIR = ROOT / "Stage_3" / "results" / FRAME / "masks"
RESULTS_DIR = ROOT / "Stage_3" / "results" / FRAME
CLOUDS_DIR = RESULTS_DIR / "lidar_car_clouds"

SUMMARY_PATH = RESULTS_DIR / f"{FRAME}_lidar_mask_association.csv"
OVERLAY_PATH = RESULTS_DIR / f"{FRAME}_lidar_mask_association.png"

# Only draw a sample of points in the overlay for readability.
MAX_DISPLAY_POINTS = 12000


def parse_calibration(path):
    """Read numeric calibration fields from a text file."""
    fields = {}
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, data = line.split(":", 1)
            values = re.findall(pattern, data)

            if values:
                fields[key.strip()] = np.asarray(
                    [float(v) for v in values],
                    dtype=np.float64,
                )

    return fields


def load_cam_to_velo(path):
    """Load the 3x4 cam0-to-Velodyne transform."""
    text = Path(path).read_text(encoding="utf-8")
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    values = np.asarray(
        [float(v) for v in re.findall(pattern, text)],
        dtype=np.float64,
    )

    if values.size != 12:
        raise ValueError(
            f"Expected 12 transform values, got {values.size}"
        )

    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def main():
    print("=" * 65)
    print("STAGE 3 - TASK 4: LIDAR-MASK ASSOCIATION")
    print("=" * 65)

    # Load camera image
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    height, width = image.shape[:2]

    # Load LiDAR points: x, y, z, reflectance
    raw = np.fromfile(LIDAR_PATH, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError("LiDAR file does not contain 4 values per point")

    lidar_all = raw.reshape(-1, 4)
    original_indices = np.arange(len(lidar_all))

    finite = np.isfinite(lidar_all[:, :3]).all(axis=1)
    lidar = lidar_all[finite]
    original_indices = original_indices[finite]

    xyz = lidar[:, :3].astype(np.float64)

    # Load calibration
    perspective = parse_calibration(PERSPECTIVE_PATH)
    R_rect = perspective["R_rect_00"].reshape(3, 3)
    P_rect = perspective["P_rect_00"].reshape(3, 4)

    T_cam_to_velo = load_cam_to_velo(CAM_TO_VELO_PATH)
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    # Transform Velodyne points to camera coordinates.
    xyz_h = np.column_stack((xyz, np.ones(len(xyz))))
    cam = (T_velo_to_cam @ xyz_h.T).T[:, :3]

    # Rectify and project.
    rect = (R_rect @ cam.T).T
    rect_h = np.column_stack((rect, np.ones(len(rect))))
    projected = (P_rect @ rect_h.T).T
    depth = projected[:, 2]

    in_front = depth > 1e-6
    pixels = np.full((len(xyz), 2), np.nan)
    pixels[in_front] = (
        projected[in_front, :2] / depth[in_front, None]
    )

    inside_image = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    # Round valid pixel coordinates for mask indexing.
    valid_point_positions = np.flatnonzero(inside_image)
    pixel_xy = np.rint(pixels[inside_image]).astype(int)
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, width - 1)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, height - 1)

    # Create output folders.
    CLOUDS_DIR.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(MASKS_DIR.glob("car_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {MASKS_DIR}")

    summary_rows = []
    overlay = image.copy()
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (128, 0, 255),
        (255, 128, 0),
        (128, 255, 0),
    ]

    # Assign projected points to each car mask.
    for mask_index, mask_path in enumerate(mask_paths):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping unreadable mask: {mask_path.name}")
            continue

        if mask.shape != (height, width):
            raise ValueError(
                f"{mask_path.name} dimensions do not match image"
            )

        binary_mask = mask > 0

        # Test whether each projected pixel belongs to this mask.
        mask_pixels = binary_mask[pixel_xy[:, 1], pixel_xy[:, 0]]
        assigned_positions = valid_point_positions[mask_pixels]

        # Keep XYZ, reflectance, and original point index.
        assigned_cloud = np.column_stack((
            lidar[assigned_positions],
            original_indices[assigned_positions],
        ))

        car_id = mask_index + 1
        cloud_path = CLOUDS_DIR / f"car_{car_id:02d}_points.npy"
        np.save(cloud_path, assigned_cloud.astype(np.float64))

        point_count = len(assigned_cloud)
        summary_rows.append({
            "frame": FRAME,
            "car_id": car_id,
            "mask_file": mask_path.name,
            "mask_pixels": int(np.count_nonzero(binary_mask)),
            "associated_lidar_points": point_count,
            "cloud_file": cloud_path.name,
        })

        color = colors[mask_index % len(colors)]

        # Draw the mask contour.
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 1)

        print(
            f"Car {car_id}: mask pixels="
            f"{int(np.count_nonzero(binary_mask))}, "
            f"associated LiDAR points={point_count}"
        )

    # Draw a sampled set of projected points for visualization.
    if len(valid_point_positions) > MAX_DISPLAY_POINTS:
        sample_ids = np.linspace(
            0, len(valid_point_positions) - 1,
            MAX_DISPLAY_POINTS, dtype=int,
        )
        display_pixels = pixel_xy[sample_ids]
    else:
        display_pixels = pixel_xy

    for u, v in display_pixels:
        cv2.circle(overlay, (int(u), int(v)), 1, (255, 255, 255), -1)

    # Save summary CSV.
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "frame", "car_id", "mask_file", "mask_pixels",
            "associated_lidar_points", "cloud_file",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    if not cv2.imwrite(str(OVERLAY_PATH), overlay):
        raise RuntimeError(f"Could not save overlay: {OVERLAY_PATH}")

    print("-" * 65)
    print(f"Finite LiDAR points: {len(lidar)}")
    print(f"Points projected inside image: {len(valid_point_positions)}")
    print(f"Car masks processed: {len(summary_rows)}")
    print(f"Point clouds saved in: {CLOUDS_DIR}")
    print(f"Summary CSV: {SUMMARY_PATH}")
    print(f"Overlay: {OVERLAY_PATH}")
    print("Task 4 completed.")


if __name__ == "__main__":
    main()
