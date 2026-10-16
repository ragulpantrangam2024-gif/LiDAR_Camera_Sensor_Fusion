
from pathlib import Path
import json
import numpy as np
import csv

ROOT = Path(__file__).resolve().parents[1]
BOX_DIR = ROOT / "bboxes_3D_cam0"
OUTPUT_DIR = ROOT / "Stage_5" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = []

for frame_file in sorted(BOX_DIR.glob("BBoxes_*.json")):
    frame_id = frame_file.stem.replace("BBoxes_", "")

    try:
        with frame_file.open("r", encoding="utf-8") as f:
            boxes = json.load(f)
    except Exception as exc:
        summary.append({
            "frame_id": frame_id,
            "boxes_found": 0,
            "valid_boxes": 0,
            "invalid_boxes": 1,
            "status": f"read_error: {exc}",
        })
        continue

    valid_count = 0
    invalid_count = 0

    for box in boxes:
        try:
            corners = np.asarray(box["corners_cam0"], dtype=float)

            valid = (
                corners.shape == (8, 3)
                and np.isfinite(corners).all()
            )

            if valid:
                valid_count += 1
            else:
                invalid_count += 1

        except (KeyError, TypeError, ValueError):
            invalid_count += 1

    summary.append({
        "frame_id": frame_id,
        "boxes_found": len(boxes),
        "valid_boxes": valid_count,
        "invalid_boxes": invalid_count,
        "status": "valid" if invalid_count == 0 else "check_required",
    })

    print(
        f"Frame {frame_id}: boxes={len(boxes)}, "
        f"valid={valid_count}, invalid={invalid_count}"
    )

csv_path = OUTPUT_DIR / "ground_truth_validation.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "frame_id", "boxes_found", "valid_boxes",
        "invalid_boxes", "status"
    ])
    writer.writeheader()
    writer.writerows(summary)

print("\nGround-truth validation completed.")
print("Frames inspected:", len(summary))
print("Total boxes:", sum(row["boxes_found"] for row in summary))
print("Valid boxes:", sum(row["valid_boxes"] for row in summary))
print("Invalid boxes:", sum(row["invalid_boxes"] for row in summary))
print("Report:", csv_path)
