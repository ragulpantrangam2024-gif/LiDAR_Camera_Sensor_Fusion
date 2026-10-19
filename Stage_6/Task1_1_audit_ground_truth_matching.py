
from pathlib import Path
import json

import pandas as pd


# ============================================================
# STAGE 6 - TASK 1.1
# Audit data for ground-truth matching
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE3_DIR = (
    ROOT / "Stage_3" / "results" / "LiDAR_Mask_Association"
)

GT_DIR = ROOT / "bboxes_3D_cam0"

TASK3_DIR = ROOT / "Stage_5" / "results" / "Task3"
TASK4_DIR = ROOT / "Stage_5" / "results" / "Task4"

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_CAR_CSV = TASK3_DIR / "per_car_evaluation.csv"
CANDIDATES_CSV = TASK3_DIR / "all_box_candidates.csv"
METRICS_CSV = TASK4_DIR / "overall_metrics.csv"


def load_csv(path):
    if not path.exists():
        print(f"\nMISSING FILE: {path}")
        return None

    df = pd.read_csv(path)

    print(f"\nFile: {path.relative_to(ROOT)}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    return df


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 1.1")
    print("GROUND-TRUTH MATCHING DATA AUDIT")
    print("=" * 72)

    per_car = load_csv(PER_CAR_CSV)
    candidates = load_csv(CANDIDATES_CSV)
    load_csv(METRICS_CSV)

    if per_car is None:
        raise FileNotFoundError(PER_CAR_CSV)

    required = {"frame_id", "car_cloud", "best_gt_index"}

    if not required.issubset(per_car.columns):
        raise ValueError(
            "Unexpected per-car CSV schema. Missing columns: "
            f"{sorted(required - set(per_car.columns))}"
        )

    # Normalize frame identifiers for grouping.
    per_car["frame_id"] = (
        per_car["frame_id"].astype(str).str.zfill(10)
    )

    print("\n" + "=" * 72)
    print("PREDICTED CLOUD SUMMARY")
    print("=" * 72)

    print("Cloud records:", len(per_car))
    print("Distinct frames:", per_car["frame_id"].nunique())

    print("\nClouds per frame:")
    print(
        per_car.groupby("frame_id")
        .size()
        .to_string()
    )

    # Count clouds selecting each ground-truth index within a frame.
    selected = per_car[
        per_car["best_gt_index"].notna()
    ].copy()

    selected["best_gt_index"] = (
        selected["best_gt_index"].astype(str)
    )

    counts = (
        selected.groupby(["frame_id", "best_gt_index"])
        .size()
        .reset_index(name="clouds_selecting_gt")
    )

    duplicates = counts[counts["clouds_selecting_gt"] > 1]

    print("\n" + "=" * 72)
    print("DUPLICATE GROUND-TRUTH CANDIDATE SELECTIONS")
    print("=" * 72)
    print("Clouds with a selected candidate:", len(selected))
    print("Distinct frame/GT selections:", len(counts))
    print("Frame/GT pairs selected by multiple clouds:", len(duplicates))

    if duplicates.empty:
        print("No duplicate selections found.")
    else:
        print(duplicates.to_string(index=False))
        duplicates.to_csv(
            OUTPUT_DIR / "duplicate_gt_selections.csv",
            index=False,
        )

    # Compare cloud records against available GT object counts.
    frame_rows = []

    for frame_id, group in per_car.groupby("frame_id"):
        gt_path = GT_DIR / f"BBoxes_{int(frame_id)}.json"

        if not gt_path.exists():
            gt_count = None
            status = "MISSING_GT_FILE"
        else:
            try:
                with open(gt_path, "r", encoding="utf-8") as file:
                    gt_records = json.load(file)

                gt_count = len(gt_records)
                status = "OK"
            except Exception as error:
                gt_count = None
                status = f"GT_READ_ERROR: {error}"

        frame_rows.append({
            "frame_id": frame_id,
            "predicted_clouds": len(group),
            "ground_truth_objects": gt_count,
            "status": status,
        })

    frame_summary = pd.DataFrame(frame_rows)
    frame_summary.to_csv(
        OUTPUT_DIR / "frame_matching_audit.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("FRAME-LEVEL MATCHING AUDIT")
    print("=" * 72)
    print(frame_summary.to_string(index=False))

    if candidates is not None:
        print("\nCandidate table available for matching-rule design.")
        print("Candidate records:", len(candidates))

        if {"frame_id", "gt_index"}.issubset(candidates.columns):
            print(
                "Candidate GT counts by frame can be compared "
                "with annotation counts."
            )
        else:
            print(
                "Inspect the printed candidate-table columns "
                "before implementing the matching algorithm."
            )

    print("\n" + "=" * 72)
    print("TASK 1.1 AUDIT COMPLETED")
    print("=" * 72)
    print("Output directory:", OUTPUT_DIR)
    print(
        "No precision, recall, F1-score, or 3D IoU has been "
        "calculated in this audit."
    )


if __name__ == "__main__":
    main()
