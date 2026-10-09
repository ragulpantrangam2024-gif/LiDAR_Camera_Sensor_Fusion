
from pathlib import Path
import json
import re

import cv2
import numpy as np

# ============================================================
# STAGE 2 - TASK 9
# Validate selected ground-truth 3D bounding boxes
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FRAME = "0000000100"
OBJECT_IDS = {145, 210, 215}

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
JSON_PATH = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"
CALIB_PATH = ROOT / "calibration" / "perspective.txt"

RESULTS_DIR = ROOT / "Stage_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    RESULTS_DIR / f"{FRAME}_task9_selected_boxes.png"
)

# Corner connections based on the existing Task 7 convention.
# Bottom face: 0-1-2-3
# Top face:    4-5-6-7
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 5), (1, 4), (2, 7), (3, 6),
]


def load_calibration(path):
    """Read R_rect_00 and P_rect_00 from perspective.txt."""
    values = {}

    number_pattern = (
        r"[-+]?(?:\d+\.?\d*|\.\d+)"
        r"(?:[eE][-+]?\d+)?"
    )

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, data = line.split(":", 1)
            numbers = re.findall(number_pattern, data)

            if numbers:
                values[key.strip()] = np.array(
                    [float(value) for value in numbers],
                    dtype=np.float64,
                )

    if "R_rect_00" not in values:
        raise ValueError("R_rect_00 not found in perspective.txt")

    if "P_rect_00" not in values:
        raise ValueError("P_rect_00 not found in perspective.txt")

    R_rect = values["R_rect_00"].reshape(3, 3)
    P_rect = values["P_rect_00"].reshape(3, 4)

    return R_rect, P_rect


def load_objects(path):
    """Load ground-truth objects from the frame JSON."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("objects", "annotations", "boxes", "bboxes"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unexpected JSON format. Check BBoxes_100.json."
    )


def project_corners(corners, R_rect, P_rect):
    """
    Project cam0 corners using the same assumption as Task 7:
    corners are in unrectified cam0 coordinates.
    """
    rectified = (R_rect @ corners.T).T

    homogeneous = np.column_stack(
        (rectified, np.ones(len(corners)))
    )

    projected = (P_rect @ homogeneous.T).T
    depth = projected[:, 2]

    pixels = np.full((len(corners), 2), np.nan)
    valid = depth > 1e-6

    pixels[valid] = (
        projected[valid, :2] / depth[valid, None]
    )

    return pixels, depth, valid


def draw_clipped_edge(image, p1, p2, color, thickness=2):
    """Draw an edge clipped to the image boundaries."""
    height, width = image.shape[:2]

    p1 = tuple(np.rint(p1).astype(int))
    p2 = tuple(np.rint(p2).astype(int))

    success, clipped_p1, clipped_p2 = cv2.clipLine(
        (0, 0, width, height), p1, p2
    )

    if success:
        cv2.line(
            image,
            clipped_p1,
            clipped_p2,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main():
    print("=" * 65)
    print("STAGE 2 - TASK 9: SELECTED BOX VALIDATION")
    print("=" * 65)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    R_rect, P_rect = load_calibration(CALIB_PATH)
    objects = load_objects(JSON_PATH)

    selected = {
        obj["index"]: obj
        for obj in objects
        if obj["index"] in OBJECT_IDS
    }

    missing_ids = OBJECT_IDS - set(selected.keys())
    if missing_ids:
        print(f"Warning: object IDs not found: {sorted(missing_ids)}")

    height, width = image.shape[:2]
    drawn_count = 0

    for object_id in sorted(selected):
        obj = selected[object_id]

        corners = np.asarray(
            obj["corners_cam0"], dtype=np.float64
        )

        if corners.shape != (8, 3):
            print(
                f"Object {object_id}: invalid corner shape "
                f"{corners.shape}"
            )
            continue

        pixels, depth, valid = project_corners(
            corners, R_rect, P_rect
        )

        inside = (
            valid
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )

        print(f"\nObject ID: {object_id}")
        print(f"Positive-depth corners: {valid.sum()}/8")
        print(f"Corners inside image: {inside.sum()}/8")

        for i in range(8):
            if valid[i]:
                u, v = pixels[i]
                print(
                    f"  Corner {i}: u={u:.1f}, v={v:.1f}, "
                    f"inside={inside[i]}"
                )
            else:
                print(f"  Corner {i}: invalid/behind camera")

        # Draw each edge only if both endpoints have positive depth.
        for start, end in EDGES:
            if valid[start] and valid[end]:
                draw_clipped_edge(
                    image,
                    pixels[start],
                    pixels[end],
                    (0, 255, 0),
                    thickness=2,
                )

        # Label visible corners with their indices.
        for i in range(8):
            if not inside[i]:
                continue

            x, y = np.rint(pixels[i]).astype(int)

            cv2.circle(
                image, (x, y), 4, (0, 0, 255), -1
            )

            cv2.putText(
                image,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Label the object near its projected center, if visible.
        visible_pixels = pixels[inside]

        if len(visible_pixels):
            center = np.mean(visible_pixels, axis=0)
            label_x = int(np.clip(center[0], 0, width - 1))
            label_y = int(np.clip(center[1], 18, height - 1))

            cv2.putText(
                image,
                f"ID {object_id}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        drawn_count += 1

    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

    print("\n" + "=" * 65)
    print(f"Selected objects processed: {drawn_count}")
    print(f"Output image: {OUTPUT_PATH}")
    print("Task 9 diagnostic completed.")
    print("No calibration or annotation data was modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
