
from pathlib import Path
import json
import re

import cv2
import numpy as np

# ============================================================
# STAGE 2 - TASK 7: VISUALIZE GROUND-TRUTH 3D BOXES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FRAME = "0000000100"

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
JSON_PATH = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"
CALIB_PATH = ROOT / "calibration" / "perspective.txt"

RESULTS_DIR = ROOT / "Stage_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / f"{FRAME}_ground_truth_boxes_overlay.png"

# Corner connectivity based on the project's box convention
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 5), (1, 4), (2, 7), (3, 6),
]


def load_calibration(path):
    values = {}

    # Extract numeric tokens robustly to avoid the previous
    # np.fromstring deprecation warning.
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, data = line.split(":", 1)
            numbers = re.findall(pattern, data)

            if numbers:
                values[key.strip()] = np.array(
                    [float(x) for x in numbers],
                    dtype=np.float64,
                )

    R_rect = values["R_rect_00"].reshape(3, 3)
    P_rect = values["P_rect_00"].reshape(3, 4)

    return R_rect, P_rect


def load_objects(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("objects", "annotations", "boxes", "bboxes"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unexpected JSON structure. Reuse the object loader "
        "from Task 6 if your JSON has a different structure."
    )


def project_corners(corners, R_rect, P_rect):
    rectified = (R_rect @ corners.T).T

    homogeneous = np.column_stack(
        (rectified, np.ones(len(rectified)))
    )

    projected = (P_rect @ homogeneous.T).T
    depth = projected[:, 2]

    pixels = np.full((len(corners), 2), np.nan)
    valid = depth > 1e-6

    pixels[valid] = (
        projected[valid, :2] / depth[valid, None]
    )

    return pixels, valid


def main():
    print("=" * 65)
    print("STAGE 2 - TASK 7: GROUND-TRUTH BOX OVERLAY")
    print("=" * 65)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    height, width = image.shape[:2]
    R_rect, P_rect = load_calibration(CALIB_PATH)
    objects = load_objects(JSON_PATH)

    drawn_boxes = 0
    skipped_boxes = 0

    for obj in objects:
        object_id = obj["index"]
        corners = np.asarray(
            obj["corners_cam0"], dtype=np.float64
        )

        if corners.shape != (8, 3):
            skipped_boxes += 1
            continue

        pixels, valid = project_corners(
            corners, R_rect, P_rect
        )

        # Draw only edges whose two endpoints have positive depth.
        drawn_edges = 0

        for start, end in EDGES:
            if not (valid[start] and valid[end]):
                continue

            p1 = np.rint(pixels[start]).astype(int)
            p2 = np.rint(pixels[end]).astype(int)

            # OpenCV clips lines at the image boundary.
            cv2.line(
                image,
                tuple(p1),
                tuple(p2),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            drawn_edges += 1

        # Label the box near the average of its visible corners.
        visible_pixels = pixels[valid]

        if len(visible_pixels):
            center = np.mean(visible_pixels, axis=0)
            x = int(np.clip(round(center[0]), 0, width - 1))
            y = int(np.clip(round(center[1]), 15, height - 1))

            cv2.putText(
                image,
                str(object_id),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if drawn_edges:
            drawn_boxes += 1
        else:
            skipped_boxes += 1

    success = cv2.imwrite(str(OUTPUT_PATH), image)
    if not success:
        raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

    print(f"Frame: {FRAME}")
    print(f"Image size: {width} x {height}")
    print(f"Objects in annotation: {len(objects)}")
    print(f"Boxes with drawable edges: {drawn_boxes}")
    print(f"Boxes without drawable edges: {skipped_boxes}")
    print(f"Saved overlay: {OUTPUT_PATH}")
    print("\nTask 7 completed.")


if __name__ == "__main__":
    main()
