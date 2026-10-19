
from pathlib import Path
import json

import pandas as pd


# ============================================================
# STAGE 6 - TASK 2
# Provisional Precision, Recall and F1-Score
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "per_car_evaluation.csv"
)

MATCHES_CSV = (
    ROOT / "Stage_6" / "results" / "Task1_5"
    / "global_one_to_one_matches.csv"
)

GT_DIR = ROOT / "bboxes_3D_cam0"

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_frames(df):
    df = df.copy()
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)
    return df


def load_ground_truth_counts():
    records = []

    for path in sorted(GT_DIR.glob("BBoxes_*.json")):
        frame_number = path.stem.replace("BBoxes_", "")
        frame_id = frame_number.zfill(10)

        with open(path, "r", encoding="utf-8") as file:
            annotations = json.load(file)

        records.append({
            "frame_id": frame_id,
            "ground_truth_records": len(annotations),
        })

    return pd.DataFrame(
        records,
        columns=["frame_id", "ground_truth_records"],
    )


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 2")
    print("PROVISIONAL PERFORMANCE METRICS")
    print("=" * 72)

    for path in [PREDICTIONS_CSV, MATCHES_CSV, GT_DIR]:
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = normalize_frames(pd.read_csv(PREDICTIONS_CSV))
    matches = normalize_frames(pd.read_csv(MATCHES_CSV))
    gt_counts = load_ground_truth_counts()

    # Verify that the global matches are one-to-one.
    duplicate_clouds = matches.duplicated(
        ["frame_id", "car_cloud"]
    ).sum()

    duplicate_boxes = matches.duplicated(
        ["frame_id", "gt_index"]
    ).sum()

    if duplicate_clouds or duplicate_boxes:
        raise ValueError(
            "Matching file violates one-to-one constraints: "
            f"duplicate clouds={duplicate_clouds}, "
            f"duplicate boxes={duplicate_boxes}"
        )

    # Every row in the Stage 5 per-car file is treated as one
    # predicted cloud for this exploratory calculation.
    predicted_count = len(
        predictions[["frame_id", "car_cloud"]].drop_duplicates()
    )

    matched_count = len(
        matches[["frame_id", "car_cloud"]].drop_duplicates()
    )

    unmatched_prediction_count = predicted_count - matched_count

    # This is only an annotation-coverage proxy, not validated
    # recall: the annotation files have no class labels.
    total_gt_records = int(gt_counts["ground_truth_records"].sum())

    matched_gt_count = len(
        matches[["frame_id", "gt_index"]].drop_duplicates()
    )

    unmatched_gt_count = total_gt_records - matched_gt_count

    # Exploratory precision-like association rate.
    association_precision = safe_divide(
        matched_count, predicted_count
    )

    # Not true recall unless every GT record is an eligible target.
    annotation_coverage_proxy = safe_divide(
        matched_gt_count, total_gt_records
    )

    # Exploratory only: combines the precision-like rate with
    # annotation coverage, under the unverified all-records-eligible
    # assumption. Do not report as validated detection F1.
    proxy_f1 = safe_divide(
        2 * association_precision * annotation_coverage_proxy,
        association_precision + annotation_coverage_proxy,
    )

    overall = pd.DataFrame([
        {
            "metric": "predicted_clouds",
            "value": predicted_count,
            "interpretation": "Number of predicted cloud records",
        },
        {
            "metric": "provisional_matches",
            "value": matched_count,
            "interpretation": "One-to-one geometric matches at 30% threshold",
        },
        {
            "metric": "unmatched_predictions",
            "value": unmatched_prediction_count,
            "interpretation": "Predicted clouds not matched at this threshold",
        },
        {
            "metric": "ground_truth_annotation_records",
            "value": total_gt_records,
            "interpretation": "All records in annotation JSON files",
        },
        {
            "metric": "matched_ground_truth_records",
            "value": matched_gt_count,
            "interpretation": "Unique GT records selected by provisional matches",
        },
        {
            "metric": "unmatched_ground_truth_records",
            "value": unmatched_gt_count,
            "interpretation": "Unmatched annotation records; not confirmed false negatives",
        },
        {
            "metric": "association_precision_proxy_percent",
            "value": association_precision * 100,
            "interpretation": "Matches / predicted clouds; provisional geometric proxy",
        },
        {
            "metric": "annotation_coverage_proxy_percent",
            "value": annotation_coverage_proxy * 100,
            "interpretation": "Matched GT records / all annotation records; not validated recall",
        },
        {
            "metric": "exploratory_f1_proxy_percent",
            "value": proxy_f1 * 100,
            "interpretation": "Exploratory combination only; not validated detection F1",
        },
    ])

    overall.to_csv(
        OUTPUT_DIR / "provisional_overall_metrics.csv",
        index=False,
    )

    # Frame-level counts, retaining frames with no predictions.
    pred_counts = (
        predictions.groupby("frame_id")["car_cloud"]
        .nunique()
        .rename("predicted_clouds")
        .reset_index()
    )

    match_counts = (
        matches.groupby("frame_id")
        .agg(
            provisional_matches=("car_cloud", "nunique"),
            matched_gt_records=("gt_index", "nunique"),
        )
        .reset_index()
    )

    per_frame = gt_counts.merge(
        pred_counts, on="frame_id", how="left"
    ).merge(
        match_counts, on="frame_id", how="left"
    )

    for column in [
        "predicted_clouds",
        "provisional_matches",
        "matched_gt_records",
    ]:
        per_frame[column] = per_frame[column].fillna(0).astype(int)

    per_frame["unmatched_predictions"] = (
        per_frame["predicted_clouds"]
        - per_frame["provisional_matches"]
    )

    per_frame["unmatched_gt_records"] = (
        per_frame["ground_truth_records"]
        - per_frame["matched_gt_records"]
    )

    per_frame["association_precision_proxy_percent"] = (
        per_frame["provisional_matches"]
        .div(per_frame["predicted_clouds"].replace(0, pd.NA))
        .mul(100)
    )

    per_frame["annotation_coverage_proxy_percent"] = (
        per_frame["matched_gt_records"]
        .div(per_frame["ground_truth_records"].replace(0, pd.NA))
        .mul(100)
    )

    per_frame.to_csv(
        OUTPUT_DIR / "provisional_per_frame_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("OVERALL EXPLORATORY RESULTS")
    print("=" * 72)

    for _, row in overall.iterrows():
        value = row["value"]
        if pd.notna(value):
            print(f"{row['metric']}: {value:.2f}")
        else:
            print(f"{row['metric']}: undefined")

    print("\nFrame-level results:")
    print(per_frame.to_string(index=False))

    print("\nOutput directory:", OUTPUT_DIR)
    print("  provisional_overall_metrics.csv")
    print("  provisional_per_frame_metrics.csv")

    print("\nIMPORTANT INTERPRETATION:")
    print("- Association precision is a provisional geometric proxy.")
    print("- Annotation coverage is NOT validated recall.")
    print("- The exploratory F1 proxy is NOT validated detection F1.")
    print(
        "- Ground-truth annotations have no class labels, and "
        "box geometry/matching assumptions remain unverified."
    )


if __name__ == "__main__":
    main()
