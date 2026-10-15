
from pathlib import Path
import csv
import re

import cv2
import numpy as np


# ============================================================
# STAGE 4 - TASK 1
# Instance-colored LiDAR projection onto camera images
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "Stage_1" / "Images"
CALIB_DIR = ROOT / "calibration"

SEG_DIR = ROOT / "Stage_3" / "results" / "YOLOv8_Segmentation"
CLOUD_DIR = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)

RESULTS_DIR = ROOT / "Stage_4" / "results"
OUTPUT_DIR = RESULTS_DIR / "Instance_Colored_Projection"

# None processes every frame with available images and clouds.
# Example: ["0000000100"]
FRAME_FILTER = None

# OpenCV colors are BGR.
COLORS = [
    (0, 0, 255),       # Red
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (0, 255, 255),     # Yellow
    (255, 0, 255),     # Magenta
    (255, 255, 0),     # Cyan
    (0, 165, 255),     # Orange
    (128, 0, 255),
]


def read_numeric_fields(path):
    fields = {}
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            values = re.findall(pattern, value)

            if values:
                fields[key.strip()] = np.array(
                    [float(v) for v in values],
                    dtype=np.float64,
                )

    return fields


def load_transform(path):
    text = Path(path).read_text(encoding="utf-8")
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    values = np.array(
        [float(v) for v in re.findall(pattern, text)],
        dtype=np.float64,
    )

    if values.size != 12:
        raise ValueError(
            f"Expected 12 transform values in {path}, "
            f"found {values.size}"
        )

    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def project_points(points, T_velo_to_cam, R_rect, P_rect,
                   width, height):
    """Project Velodyne XYZ points into the camera image."""

    if len(points) == 0:
        return np.empty((0, 2), dtype=int)

    xyz = points[:, :3].astype(np.float64)
    xyz_h = np.column_stack((xyz, np.ones(len(xyz))))

    cam = (T_velo_to_cam @ xyz_h.T).T[:, :3]
    rect = (R_rect @ cam.T).T
    rect_h = np.column_stack((rect, np.ones(len(rect))))

    projected = (P_rect @ rect_h.T).T
    depth = projected[:, 2]

    valid_depth = depth > 1e-6

    pixels = np.full((len(points), 2), np.nan)
    pixels[valid_depth] = (
        projected[valid_depth, :2]
        / depth[valid_depth, None]
    )

    inside = (
        valid_depth
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    xy = np.rint(pixels[inside]).astype(int)

    if len(xy):
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)

    return xy


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load camera calibration.
    perspective = read_numeric_fields(
        CALIB_DIR / "perspective.txt"
    )

    R_rect = perspective["R_rect_00"].reshape(3, 3)
    P_rect = perspective["P_rect_00"].reshape(3, 4)

    T_cam_to_velo = load_transform(
        CALIB_DIR / "calib_cam_to_velo.txt"
    )
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if FRAME_FILTER is not None:
        image_paths = [
            path for path in image_paths
            if path.stem in FRAME_FILTER
        ]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    summary_rows = []

    for image_path in image_paths:
        frame = image_path.stem

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping {frame}: image could not be read")
            continue

        height, width = image.shape[:2]

        frame_cloud_dir = (
            CLOUD_DIR / frame / "car_point_clouds"
        )

        if not frame_cloud_dir.exists():
            print(f"Skipping {frame}: point-cloud folder missing")
            continue

        cloud_paths = sorted(
            frame_cloud_dir.glob("car_*_points.npy")
        )

        overlay = image.copy()
        total_points = 0
        visible_points = 0
        car_count = 0

        for cloud_path in cloud_paths:
            cloud = np.load(cloud_path)

            if cloud.ndim != 2 or cloud.shape[1] < 3:
                print(f"Skipping invalid cloud: {cloud_path.name}")
                continue

            car_count += 1
            color = COLORS[(car_count - 1) % len(COLORS)]

            total_points += len(cloud)

            pixels = project_points(
                cloud,
                T_velo_to_cam,
                R_rect,
                P_rect,
                width,
                height,
            )

            visible_points += len(pixels)

            # Draw each associated LiDAR point using its car color.
            for u, v in pixels:
                cv2.circle(
                    overlay,
                    (int(u), int(v)),
                    2,
                    color,
                    -1,
                )

            # Label each car instance.
            cv2.putText(
                overlay,
                f"Car {car_count}: {len(cloud)} points",
                (15, 25 + (car_count - 1) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        output_path = (
            OUTPUT_DIR / f"{frame}_instance_colored_lidar.png"
        )

        cv2.imwrite(str(output_path), overlay)

        summary_rows.append({
            "frame": frame,
            "car_clouds": car_count,
            "associated_points": total_points,
            "projected_points_in_image": visible_points,
            "output_image": output_path.name,
        })

        print({
            "frame": frame,
            "car_clouds": car_count,
            "associated_points": total_points,
            "projected_points_in_image": visible_points,
            "output": str(output_path),
        })

    summary_path = OUTPUT_DIR / "projection_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "frame",
            "car_clouds",
            "associated_points",
            "projected_points_in_image",
            "output_image",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nStage 4 Task 1 completed.")
    print(f"Frames processed: {len(summary_rows)}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
