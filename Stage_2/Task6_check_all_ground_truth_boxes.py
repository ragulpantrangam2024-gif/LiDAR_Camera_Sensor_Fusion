
from pathlib import Path
import json
import csv
import numpy as np
import cv2

# ============================================================
# STAGE 2 - TASK 6: CHECK ALL GROUND-TRUTH BOXES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FRAME = "0000000100"

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
JSON_PATH = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"
CALIB_PATH = ROOT / "calibration" / "perspective.txt"

RESULTS_DIR = ROOT / "Stage_2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / f"{FRAME}_all_boxes_diagnostic.csv"


def load_calibration(path):
    values = {}

    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, data = line.split(":", 1)

            try:
                numbers = np.fromstring(data, sep=" ")
                if numbers.size:
                    values[key.strip()] = numbers
            except ValueError:
                continue

    R_rect = values["R_rect_00"].reshape(3, 3)
    P_rect = values["P_rect_00"].reshape(3, 4)

    return R_rect, P_rect


def load_objects(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Supports a list or a dictionary containing an object list
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("objects", "annotations", "boxes", "bboxes"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unknown JSON structure. Check the Task 4 JSON loader."
    )


def project_corners(corners, R_rect, P_rect):
    # Assumption being tested:
    # corners_cam0 are in unrectified camera-0 coordinates.

    rectified = (R_rect @ corners.T).T

    homogeneous = np.column_stack([
        rectified,
        np.ones(len(rectified))
    ])

    projected = (P_rect @ homogeneous.T).T

    depth = projected[:, 2]

    pixels = np.full((len(corners), 2), np.nan)

    valid = depth > 1e-6

    pixels[valid] = (
        projected[valid, :2] /
        depth[valid, None]
    )

    return pixels, valid


def main():
    print("=" * 65)
    print("STAGE 2 - TASK 6: ALL GROUND-TRUTH BOX DIAGNOSTICS")
    print("=" * 65)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    height, width = image.shape[:2]

    R_rect, P_rect = load_calibration(CALIB_PATH)
    objects = load_objects(JSON_PATH)

    results = []

    print(f"Frame: {FRAME}")
    print(f"Image size: {width} x {height}")
    print(f"Total objects: {len(objects)}")
    print("-" * 65)

    for obj in objects:
        object_id = obj["index"]

        corners = np.asarray(
            obj["corners_cam0"],
            dtype=np.float64
        )

        if corners.shape != (8, 3):
            print(f"Object {object_id}: invalid corner shape")
            continue

        pixels, valid = project_corners(
            corners, R_rect, P_rect
        )

        inside = (
            valid
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )

        valid_pixels = pixels[valid]

        if len(valid_pixels):
            u_min = float(np.min(valid_pixels[:, 0]))
            u_max = float(np.max(valid_pixels[:, 0]))
            v_min = float(np.min(valid_pixels[:, 1]))
            v_max = float(np.max(valid_pixels[:, 1]))

            # Axis-aligned 2D extent overlaps the image.
            # This is not an exact 3D visibility test.
            overlaps = (
                u_max >= 0
                and u_min < width
                and v_max >= 0
                and v_min < height
            )
        else:
            u_min = u_max = np.nan
            v_min = v_max = np.nan
            overlaps = False

        result = {
            "object_id": object_id,
            "positive_depth_corners": int(valid.sum()),
            "inside_image_corners": int(inside.sum()),
            "u_min": round(u_min, 2),
            "u_max": round(u_max, 2),
            "v_min": round(v_min, 2),
            "v_max": round(v_max, 2),
            "bbox_overlaps_image": overlaps,
        }

        results.append(result)

        print(
            f"Object {object_id}: "
            f"depth={valid.sum()}/8, "
            f"inside={inside.sum()}/8, "
            f"u=[{u_min:.1f}, {u_max:.1f}], "
            f"v=[{v_min:.1f}, {v_max:.1f}], "
            f"overlap={overlaps}"
        )

    if results:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

    any_inside = sum(
        r["inside_image_corners"] > 0 for r in results
    )

    overlapping = sum(
        r["bbox_overlaps_image"] for r in results
    )

    fully_inside = sum(
        r["inside_image_corners"] == 8 for r in results
    )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Objects evaluated: {len(results)}")
    print(f"Objects with at least 1 corner inside: {any_inside}")
    print(f"Objects with projected extent overlapping: {overlapping}")
    print(f"Objects with all 8 corners inside: {fully_inside}")
    print(f"\nCSV saved to: {CSV_PATH}")
    print("\nTask 6 completed.")


if __name__ == "__main__":
    main()
