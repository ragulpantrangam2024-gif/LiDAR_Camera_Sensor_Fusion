
from pathlib import Path
import json
import re

import cv2
import numpy as np

# ============================================================
# STAGE 2 - TASK 10
# Compare projected LiDAR points with ground-truth boxes
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FRAME = "0000000100"
OBJECT_IDS = {145, 210, 215}

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
LIDAR_PATH = (
    ROOT / "data_3d_raw" / "2013_05_28_drive_0000_sync"
    / "velodyne_points" / "data" / f"{FRAME}.bin"
)
JSON_PATH = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"
CAM_TO_VELO_PATH = ROOT / "calibration" / "calib_cam_to_velo.txt"
PERSPECTIVE_PATH = ROOT / "calibration" / "perspective.txt"

RESULTS_DIR = ROOT / "Stage_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / f"{FRAME}_task10_lidar_boxes.png"

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 5), (1, 4), (2, 7), (3, 6),
]


def parse_numeric_file(path):
    """Read calibration values from colon-separated text."""
    values = {}
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, data = line.split(":", 1)
            nums = re.findall(pattern, data)
            if nums:
                values[key.strip()] = np.array(
                    [float(x) for x in nums], dtype=np.float64
                )

    return values


def load_cam_to_velo(path):
    """Read the 3x4 cam0-to-Velodyne transform."""
    text = Path(path).read_text(encoding="utf-8")
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    nums = np.array(
        [float(x) for x in re.findall(pattern, text)],
        dtype=np.float64,
    )

    if nums.size != 12:
        raise ValueError(
            f"Expected 12 transform values in {path}, got {nums.size}"
        )

    T = np.eye(4)
    T[:3, :] = nums.reshape(3, 4)
    return T


def load_objects(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("objects", "annotations", "boxes", "bboxes"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Unrecognized ground-truth JSON structure")


def main():
    print("=" * 65)
    print("STAGE 2 - TASK 10: LIDAR AND BOX COMPARISON")
    print("=" * 65)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    height, width = image.shape[:2]

    # Read Velodyne binary: x, y, z, reflectance per point.
    raw = np.fromfile(LIDAR_PATH, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("LiDAR file size is not divisible by four")

    lidar = raw.reshape(-1, 4)
    xyz = lidar[:, :3].astype(np.float64)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    perspective = parse_numeric_file(PERSPECTIVE_PATH)
    R_rect = perspective["R_rect_00"].reshape(3, 3)
    P_rect = perspective["P_rect_00"].reshape(3, 4)
    T_cam_to_velo = load_cam_to_velo(CAM_TO_VELO_PATH)

    # Existing projection assumption: calibration transform is
    # cam0 -> Velodyne, so invert it to obtain Velodyne -> cam0.
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    xyz_h = np.column_stack((xyz, np.ones(len(xyz))))
    cam = (T_velo_to_cam @ xyz_h.T).T[:, :3]
    rect = (R_rect @ cam.T).T

    rect_h = np.column_stack((rect, np.ones(len(rect))))
    proj = (P_rect @ rect_h.T).T
    depth = proj[:, 2]

    in_front = depth > 1e-6
    pixels = np.full((len(xyz), 2), np.nan)
    pixels[in_front] = (
        proj[in_front, :2] / depth[in_front, None]
    )

    inside = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    # Draw depth-coloured points, sampling for readability.
    valid_indices = np.flatnonzero(inside)
    max_points = 12000

    if len(valid_indices) > max_points:
        selected = np.linspace(
            0, len(valid_indices) - 1, max_points, dtype=int
        )
        valid_indices = valid_indices[selected]

    point_depths = depth[valid_indices]
    dmin = float(point_depths.min()) if len(point_depths) else 0.0
    dmax = float(point_depths.max()) if len(point_depths) else 1.0

    for idx in valid_indices:
        u, v = np.rint(pixels[idx]).astype(int)
        ratio = (depth[idx] - dmin) / max(dmax - dmin, 1e-6)

        # Near points are red; farther points are blue.
        color = (
            int(255 * ratio),
            int(255 * (1.0 - abs(2 * ratio - 1.0))),
            int(255 * (1.0 - ratio)),
        )
        cv2.circle(image, (u, v), 1, color, -1)

    # Draw selected ground-truth boxes using the existing
    # cam0 -> rectified camera projection assumption.
    objects = load_objects(JSON_PATH)
    selected_objects = [
        obj for obj in objects if obj["index"] in OBJECT_IDS
    ]

    for obj in selected_objects:
        object_id = obj["index"]
        corners = np.asarray(obj["corners_cam0"], dtype=np.float64)

        if corners.shape != (8, 3):
            print(f"Skipping object {object_id}: invalid corner shape")
            continue

        rect_corners = (R_rect @ corners.T).T
        rect_h = np.column_stack(
            (rect_corners, np.ones(8))
        )
        box_proj = (P_rect @ rect_h.T).T
        box_depth = box_proj[:, 2]

        box_pixels = np.full((8, 2), np.nan)
        valid_corners = box_depth > 1e-6
        box_pixels[valid_corners] = (
            box_proj[valid_corners, :2]
            / box_depth[valid_corners, None]
        )

        for a, b in EDGES:
            if valid_corners[a] and valid_corners[b]:
                p1 = tuple(np.rint(box_pixels[a]).astype(int))
                p2 = tuple(np.rint(box_pixels[b]).astype(int))

                ok, c1, c2 = cv2.clipLine(
                    (0, 0, width, height), p1, p2
                )
                if ok:
                    cv2.line(
                        image, c1, c2, (0, 255, 0), 2,
                        cv2.LINE_AA
                    )

        visible = box_pixels[
            valid_corners
            & (box_pixels[:, 0] >= 0)
            & (box_pixels[:, 0] < width)
            & (box_pixels[:, 1] >= 0)
            & (box_pixels[:, 1] < height)
        ]

        if len(visible):
            center = np.mean(visible, axis=0).astype(int)
            cv2.putText(
                image, f"ID {object_id}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 255), 2, cv2.LINE_AA
            )

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Failed to save {OUTPUT_PATH}")

    print(f"Image size: {width} x {height}")
    print(f"Finite LiDAR points: {len(xyz)}")
    print(f"Points in front of camera: {int(in_front.sum())}")
    print(f"Points inside image: {int(inside.sum())}")
    print(f"LiDAR points drawn: {len(valid_indices)}")
    print(f"Selected boxes found: {len(selected_objects)}")
    print(f"Saved image: {OUTPUT_PATH}")
    print("Task 10 completed.")


if __name__ == "__main__":
    main()
