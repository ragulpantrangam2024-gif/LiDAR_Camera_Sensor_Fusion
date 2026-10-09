
from pathlib import Path
import json

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAME_ID = "0000000100"
OBJECT_ID = 111

IMAGE_PATH = PROJECT_DIR / "Stage_1" / "Images" / f"{FRAME_ID}.png"
GT_PATH = PROJECT_DIR / "bboxes_3D_cam0" / "BBoxes_100.json"
CALIB_PATH = PROJECT_DIR / "calibration" / "perspective.txt"
RESULTS_DIR = PROJECT_DIR / "Stage_2" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Edges based on the corner ordering documented for this dataset.
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 5), (1, 4), (2, 7), (3, 6),
]


def load_calibration(path):
    params = {}

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

            params[key.strip()] = values

    return params


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    height, width = image.shape[:2]

    with GT_PATH.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    obj = next(
        (item for item in annotations
         if item.get("index") == OBJECT_ID),
        None,
    )

    if obj is None:
        raise ValueError(f"Object {OBJECT_ID} not found in {GT_PATH.name}")

    corners = np.asarray(obj["corners_cam0"], dtype=np.float64)

    if corners.shape != (8, 3):
        raise ValueError(f"Expected 8x3 corners, got {corners.shape}")

    calibration = load_calibration(CALIB_PATH)
    R_rect = calibration["R_rect_00"].reshape(3, 3)
    P_rect = calibration["P_rect_00"].reshape(3, 4)

    # Assumption under test: annotation corners are in unrectified cam0.
    corners_rect = (R_rect @ corners.T).T
    corners_h = np.column_stack(
        [corners_rect, np.ones(8)]
    )
    projected = (P_rect @ corners_h.T).T

    depth = projected[:, 2]
    positive_depth = depth > 0

    uv = np.full((8, 2), np.nan, dtype=np.float64)
    uv[positive_depth, 0] = (
        projected[positive_depth, 0] / depth[positive_depth]
    )
    uv[positive_depth, 1] = (
        projected[positive_depth, 1] / depth[positive_depth]
    )

    inside = (
        positive_depth
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )

    print("=" * 60)
    print("STAGE 2 — GROUND-TRUTH BOX PROJECTION TEST")
    print("=" * 60)
    print(f"Frame: {FRAME_ID}")
    print(f"Object ID: {OBJECT_ID}")
    print(f"Image size: {width} x {height}")
    print(f"Corners with positive depth: {positive_depth.sum()}/8")
    print(f"Corners inside image: {inside.sum()}/8")
    print("\nCorner projections:")

    for i, (point, pixel) in enumerate(zip(corners, uv)):
        print(
            f"Corner {i}: cam0={np.round(point, 3)}, "
            f"pixel={np.round(pixel, 2)}, "
            f"inside={bool(inside[i])}"
        )

    # Draw only if all eight projected corners are valid.
    if inside.all():
        for a, b in EDGES:
            p1 = tuple(np.round(uv[a]).astype(int))
            p2 = tuple(np.round(uv[b]).astype(int))
            cv2.line(image, p1, p2, (0, 255, 0), 2)

        cv2.putText(
            image, f"GT object {OBJECT_ID}",
            tuple(np.round(uv[0]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        output_path = RESULTS_DIR / f"{FRAME_ID}_gt_box_{OBJECT_ID}.png"
        cv2.imwrite(str(output_path), image)
        print(f"\nBox overlay saved: {output_path}")
    else:
        print(
            "\nNo box overlay drawn: not all corners fall inside the image."
        )
        print(
            "This may indicate the object is outside the image or that "
            "the coordinate-frame assumption needs verification."
        )

    print("\nProjection test completed.")


if __name__ == "__main__":
    main()
