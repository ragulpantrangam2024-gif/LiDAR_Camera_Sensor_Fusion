
from pathlib import Path
import json

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAME_ID = "0000000100"

IMAGE_PATH = PROJECT_DIR / "Stage_1" / "Images" / f"{FRAME_ID}.png"
GT_PATH = PROJECT_DIR / "bboxes_3D_cam0" / "BBoxes_100.json"
CALIB_PATH = PROJECT_DIR / "calibration" / "perspective.txt"
RESULTS_DIR = PROJECT_DIR / "Stage_2" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

            if values.size:
                params[key.strip()] = values

    return params


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not GT_PATH.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {GT_PATH}")

    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIB_PATH}")

    image = cv2.imread(str(IMAGE_PATH))
    height, width = image.shape[:2]

    with GT_PATH.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    calibration = load_calibration(CALIB_PATH)

    P_rect = calibration["P_rect_00"].reshape(3, 4)
    R_rect = calibration["R_rect_00"].reshape(3, 3)
    rect_size = calibration["S_rect_00"].astype(int)

    print("=" * 60)
    print("STAGE 2 — TASK 4: VALIDATE BOX COORDINATES")
    print("=" * 60)
    print(f"Frame: {FRAME_ID}")
    print(f"Image dimensions: {width} x {height}")
    print(f"Rectified dimensions: {rect_size[0]} x {rect_size[1]}")
    print(f"Ground-truth objects: {len(annotations)}")

    all_corners = []

    for obj in annotations:
        corners = np.asarray(
            obj.get("corners_cam0", []),
            dtype=np.float64,
        )

        if corners.shape != (8, 3):
            print(f"Skipping invalid object: {obj.get('index')}")
            continue

        all_corners.append(corners)

    if not all_corners:
        raise ValueError("No valid 3D bounding boxes found.")

    corners_all = np.concatenate(all_corners, axis=0)

    print("\nGround-truth corner coordinate ranges:")
    for axis, name in enumerate(["X", "Y", "Z"]):
        print(
            f"{name}: {corners_all[:, axis].min():.3f} "
            f"to {corners_all[:, axis].max():.3f}"
        )

    print("\nCalibration information:")
    print("P_rect_00:")
    print(P_rect)
    print("\nR_rect_00:")
    print(R_rect)

    print("\nCoordinate validation notes:")
    print("- Ground-truth boxes contain eight corners per object.")
    print("- The annotation field is named corners_cam0.")
    print("- Coordinate ranges alone do not prove the axis convention.")
    print("- Do not project these corners until their coordinate frame")
    print("  and axis convention are confirmed from the dataset documentation.")

    # Save a diagnostic image with the image frame and dimensions.
    diagnostic = image.copy()

    cv2.putText(
        diagnostic,
        f"Frame {FRAME_ID} | {len(all_corners)} valid 3D boxes",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    output_path = RESULTS_DIR / f"{FRAME_ID}_box_coordinate_diagnostic.png"
    cv2.imwrite(str(output_path), diagnostic)

    print(f"\nDiagnostic image saved to: {output_path}")
    print("\nTask 4 completed: coordinate inspection finished.")


if __name__ == "__main__":
    main()
