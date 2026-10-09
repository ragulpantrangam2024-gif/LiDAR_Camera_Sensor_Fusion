
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
STAGE_DIR = Path(__file__).resolve().parent
GT_DIR = PROJECT_DIR / "bboxes_3D_cam0"
RESULTS_DIR = STAGE_DIR / "results"

FRAME_ID = "0000000100"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Locate the annotation file for this frame.
    json_path = GT_DIR / f"BBoxes_{int(FRAME_ID)}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    if not isinstance(annotations, list):
        raise ValueError("Expected a list of ground-truth objects.")

    valid_objects = []
    invalid_objects = []

    for obj in annotations:
        object_id = obj.get("index")
        corners = np.asarray(obj.get("corners_cam0"), dtype=np.float64)

        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            invalid_objects.append(object_id)
            continue

        valid_objects.append({
            "index": object_id,
            "corners_cam0": corners.tolist(),
            "min_corner": corners.min(axis=0).tolist(),
            "max_corner": corners.max(axis=0).tolist(),
        })

    print("=" * 60)
    print("STAGE 1 — TASK 4: GROUND-TRUTH 3D BOXES")
    print("=" * 60)
    print(f"Annotation file: {json_path.name}")
    print(f"Total objects: {len(annotations)}")
    print(f"Valid 3D boxes: {len(valid_objects)}")
    print(f"Invalid objects: {len(invalid_objects)}")

    if invalid_objects:
        print(f"Invalid object indices: {invalid_objects}")

    for obj in valid_objects[:5]:
        print(f"\nObject index: {obj['index']}")
        print(f"Minimum XYZ: {np.round(obj['min_corner'], 3)}")
        print(f"Maximum XYZ: {np.round(obj['max_corner'], 3)}")
        print(f"Corner count: {len(obj['corners_cam0'])}")

    # Save validated boxes with their original eight corners.
    json_output = RESULTS_DIR / f"{FRAME_ID}_ground_truth_validated.json"

    with json_output.open("w", encoding="utf-8") as file:
        json.dump(valid_objects, file, indent=2)

    # Save a convenient summary for inspection.
    csv_output = RESULTS_DIR / f"{FRAME_ID}_ground_truth_summary.csv"

    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "object_index",
            "corner_count",
            "min_x", "min_y", "min_z",
            "max_x", "max_y", "max_z",
        ])

        for obj in valid_objects:
            minimum = obj["min_corner"]
            maximum = obj["max_corner"]

            writer.writerow([
                obj["index"], 8,
                *minimum,
                *maximum,
            ])

    print(f"\nValidated annotations saved to: {json_output}")
    print(f"Summary CSV saved to: {csv_output}")
    print("\nTask 4 completed successfully.")


if __name__ == "__main__":
    main()
