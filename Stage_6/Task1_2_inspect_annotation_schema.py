
from pathlib import Path
from collections import Counter
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "bboxes_3D_cam0"
CLOUD_DIR = (
    ROOT / "Stage_3" / "results" / "LiDAR_Mask_Association"
)
OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_IDS = [
    "0000000100", "0000000250", "0000000360",
    "0000000570", "0000000644", "0000000729",
    "0000000947", "0000001134", "0000001280",
    "0000001338", "0000001461", "0000001597",
    "0000001791", "0000001866", "0000002033",
    "0000002098", "0000002449", "0000002717",
    "0000002903", "0000002939",
]

POSSIBLE_CLASS_KEYS = {
    "class", "label", "type", "category",
    "object_type", "name", "semantic_class",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 1.2")
    print("GROUND-TRUTH ANNOTATION SCHEMA AUDIT")
    print("=" * 72)

    summary = []
    key_counts = Counter()
    class_values = Counter()
    samples = {}

    for frame_id in FRAME_IDS:
        path = GT_DIR / f"BBoxes_{int(frame_id)}.json"

        if not path.exists():
            summary.append({
                "frame_id": frame_id,
                "gt_records": None,
                "records_with_8x3_corners": None,
                "possible_class_fields": None,
                "status": "MISSING_FILE",
            })
            print(f"Missing annotation: {path.name}")
            continue

        records = read_json(path)

        if not isinstance(records, list):
            print(
                f"Unexpected top-level type in {path.name}: "
                f"{type(records).__name__}"
            )
            summary.append({
                "frame_id": frame_id,
                "gt_records": None,
                "records_with_8x3_corners": None,
                "possible_class_fields": None,
                "status": "UNEXPECTED_TOP_LEVEL",
            })
            continue

        valid_corner_count = 0
        frame_class_counts = Counter()
        frame_class_keys = set()

        for record in records:
            if not isinstance(record, dict):
                continue

            key_counts.update(record.keys())

            corners = record.get("corners_cam0")
            try:
                if len(corners) == 8 and all(
                    len(corner) == 3 for corner in corners
                ):
                    valid_corner_count += 1
            except (TypeError, ValueError):
                pass

            for key in POSSIBLE_CLASS_KEYS:
                if key in record:
                    frame_class_keys.add(key)
                    value = str(record[key])
                    class_values[(key, value)] += 1
                    frame_class_counts[(key, value)] += 1

            if len(samples) < 5:
                samples.setdefault(frame_id, record)

        summary.append({
            "frame_id": frame_id,
            "gt_records": len(records),
            "records_with_8x3_corners": valid_corner_count,
            "possible_class_fields": ",".join(
                sorted(frame_class_keys)
            ),
            "status": "OK",
        })

        print(
            f"Frame {frame_id}: records={len(records)}, "
            f"valid corner arrays={valid_corner_count}, "
            f"possible class keys={sorted(frame_class_keys)}"
        )

        if len(records) >= 150:
            print("  Example record keys:", (
                sorted(records[0].keys()) if records else []
            ))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        OUTPUT_DIR / "annotation_schema_summary.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("ANNOTATION FIELD FREQUENCIES")
    print("=" * 72)

    for key, count in key_counts.most_common():
        print(f"{key}: {count}")

    print("\n" + "=" * 72)
    print("POSSIBLE CLASS / CATEGORY VALUES")
    print("=" * 72)

    if class_values:
        for (key, value), count in class_values.most_common(50):
            print(f"{key}={value}: {count}")
    else:
        print("No obvious class/category fields found.")

    print("\n" + "=" * 72)
    print("SAMPLE ANNOTATION RECORDS")
    print("=" * 72)

    for frame_id, record in samples.items():
        print(f"\nFrame {frame_id}:")
        print(json.dumps(record, indent=2)[:2500])

    print("\n" + "=" * 72)
    print("FRAMES WITH CLOUD OUTPUTS")
    print("=" * 72)

    for frame_id in FRAME_IDS:
        frame_dir = CLOUD_DIR / frame_id / "car_point_clouds"
        clouds = list(frame_dir.glob("car_*_points.npy"))
        print(f"{frame_id}: {len(clouds)} cloud files")

    print("\nSummary saved to:", OUTPUT_DIR)
    print("No source annotations or previous results were modified.")


if __name__ == "__main__":
    main()
