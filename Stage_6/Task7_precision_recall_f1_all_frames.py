"""
STAGE 6 - TASK 7
Precision, Recall and F1 - All Frames

Uses:
- Stage 5 Task 3 per_car_evaluation.csv: all 82 predicted LiDAR clouds
- Stage 6 Task 1.5 global_one_to_one_matches.csv: one-to-one GT matches
- Ground-truth JSON files in bboxes_3D_cam0

Classification:
- TP: matched prediction with inside ratio >= 50%
- FP: unmatched prediction OR matched prediction with inside ratio < 50%
- FN: ground-truth box not matched by a TP

Note:
These are provisional project metrics. GT annotations have no class labels,
and the geometric evaluation assumptions have not been independently validated.
"""

from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

PREDICTIONS_FILE = (
    ROOT / "Stage_5" / "results" / "Task3" / "per_car_evaluation.csv"
)

MATCHES_FILE = (
    ROOT
    / "Stage_6"
    / "results"
    / "Task1"
    / "Task1_5"
    / "global_one_to_one_matches.csv"
)

GROUND_TRUTH_DIR = ROOT / "bboxes_3D_cam0"

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task7"

TP_THRESHOLD_PERCENT = 50.0

EXPECTED_FRAMES = [
    100, 250, 360, 570, 644, 729, 947, 1134, 1280, 1338,
    1461, 1597, 1791, 1866, 2033, 2098, 2449, 2717, 2903, 2939
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def stop_with_error(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def normalize_frame_id(value):
    """Convert a frame identifier such as '0000000250' or 250 to integer 250."""
    try:
        return int(str(value).strip().replace(".0", ""))
    except (TypeError, ValueError):
        stop_with_error(f"Cannot interpret frame ID: {value!r}")


def normalize_cloud_name(value):
    """Normalize a cloud name to the filename format used by the prediction CSV."""
    name = str(value).strip()

    if not name.endswith(".npy"):
        name = f"{name}.npy"

    return name


def find_column(df, candidates, description, required=True):
    """Find a column using known possible column-name variants."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    if required:
        stop_with_error(
            f"Could not find the {description} column.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Accepted names: {candidates}"
        )

    return None


def load_ground_truth_count(frame_id):
    """
    Count ground-truth records in the corresponding JSON file.

    The current annotation files contain records with an 'index' and
    'corners_cam0'. They do not contain class labels.
    """
    frame_number = normalize_frame_id(frame_id)
    gt_file = GROUND_TRUTH_DIR / f"BBoxes_{frame_number}.json"

    if not gt_file.exists():
        stop_with_error(f"Ground-truth file not found: {gt_file}")

    try:
        with gt_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        stop_with_error(f"Could not read {gt_file}: {exc}")

    if not isinstance(data, list):
        stop_with_error(
            f"Expected a list of ground-truth records in {gt_file}"
        )

    return len(data), data


def get_gt_indices(gt_records):
    """Return ground-truth record IDs, preserving their original JSON values."""
    indices = []

    for position, record in enumerate(gt_records):
        if not isinstance(record, dict):
            continue

        if "index" in record:
            indices.append(str(record["index"]))
        else:
            indices.append(str(position))

    return set(indices)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_metrics(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision_percent": precision * 100.0,
        "recall_percent": recall * 100.0,
        "f1_percent": f1 * 100.0,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 6 - TASK 7")
    print("PRECISION, RECALL AND F1 - ALL FRAMES")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load prediction records
    # --------------------------------------------------------

    if not PREDICTIONS_FILE.exists():
        stop_with_error(f"Prediction CSV not found: {PREDICTIONS_FILE}")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions.columns = predictions.columns.str.strip()

    if predictions.empty:
        stop_with_error("The prediction CSV contains no rows.")

    prediction_frame_col = find_column(
        predictions,
        ["frame_id", "frame", "frame_number"],
        "prediction frame ID"
    )
    prediction_cloud_col = find_column(
        predictions,
        ["car_cloud", "cloud_name", "prediction", "cloud"],
        "prediction cloud identifier"
    )
    ratio_col = find_column(
        predictions,
        [
            "inside_ratio_percent",
            "containment_ratio_percent",
            "inside_percentage",
            "accuracy_percent",
        ],
        "inside-ratio percentage"
    )

    predictions = predictions.copy()

    predictions["frame_id"] = predictions[prediction_frame_col].apply(
        normalize_frame_id
    )
    predictions["car_cloud"] = predictions[prediction_cloud_col].apply(
        normalize_cloud_name
    )
    predictions["inside_ratio_percent"] = pd.to_numeric(
        predictions[ratio_col], errors="coerce"
    )

    # A prediction should be uniquely identified by frame and cloud.
    duplicate_predictions = predictions.duplicated(
        subset=["frame_id", "car_cloud"], keep=False
    )

    if duplicate_predictions.any():
        examples = predictions.loc[
            duplicate_predictions, ["frame_id", "car_cloud"]
        ].head(10)

        stop_with_error(
            "Duplicate prediction identifiers were found. Examples:\n"
            + examples.to_string(index=False)
        )

    print(f"\nPrediction rows loaded: {len(predictions)}")
    print(f"Prediction source: {PREDICTIONS_FILE}")

    # --------------------------------------------------------
    # 2. Load global one-to-one matching results
    # --------------------------------------------------------

    if not MATCHES_FILE.exists():
        stop_with_error(f"Matching CSV not found: {MATCHES_FILE}")

    matches = pd.read_csv(MATCHES_FILE)
    matches.columns = matches.columns.str.strip()

    if matches.empty:
        stop_with_error("The global matching CSV contains no rows.")

    match_frame_col = find_column(
        matches,
        ["frame_id", "frame", "frame_number"],
        "matching frame ID"
    )
    match_cloud_col = find_column(
        matches,
        ["car_cloud", "cloud_name", "prediction", "cloud"],
        "matching cloud identifier"
    )
    match_gt_col = find_column(
        matches,
        [
            "matched_gt_index",
            "gt_index",
            "best_gt_index",
            "ground_truth_index",
            "matched_ground_truth_index",
            "gt_id",
        ],
        "matched ground-truth index"
    )

    matches = matches.copy()
    matches["frame_id"] = matches[match_frame_col].apply(normalize_frame_id)
    matches["car_cloud"] = matches[match_cloud_col].apply(normalize_cloud_name)
    matches["matched_gt_index"] = matches[match_gt_col].astype(str)

    duplicate_match_predictions = matches.duplicated(
        subset=["frame_id", "car_cloud"], keep=False
    )

    if duplicate_match_predictions.any():
        examples = matches.loc[
            duplicate_match_predictions,
            ["frame_id", "car_cloud", "matched_gt_index"]
        ].head(10)

        stop_with_error(
            "The matching file contains multiple matches for a prediction. "
            "Expected one-to-one matching.\n"
            + examples.to_string(index=False)
        )

    duplicate_gt_matches = matches.duplicated(
        subset=["frame_id", "matched_gt_index"], keep=False
    )

    if duplicate_gt_matches.any():
        examples = matches.loc[
            duplicate_gt_matches,
            ["frame_id", "car_cloud", "matched_gt_index"]
        ].head(10)

        stop_with_error(
            "A ground-truth record is assigned to multiple predictions "
            "in the matching file.\n"
            + examples.to_string(index=False)
        )

    print(f"Global matching rows loaded: {len(matches)}")
    print(f"Matching source: {MATCHES_FILE}")

    # --------------------------------------------------------
    # 3. Merge matches into predictions
    # --------------------------------------------------------
    # Only bring across the identifiers. This prevents duplicate
    # inside-ratio columns such as inside_ratio_percent_x / _y.

    matches_for_merge = matches[
        ["frame_id", "car_cloud", "matched_gt_index"]
    ].copy()

    all_detections = predictions.merge(
        matches_for_merge,
        on=["frame_id", "car_cloud"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    all_detections["has_gt_match"] = (
        all_detections["_merge"] == "both"
    )

    all_detections.drop(columns=["_merge"], inplace=True)

    print(f"\nTotal predicted clouds evaluated: {len(all_detections)}")
    print(f"Matched predictions: {int(all_detections['has_gt_match'].sum())}")
    print(f"Unmatched predictions: {int((~all_detections['has_gt_match']).sum())}")

    # --------------------------------------------------------
    # 4. Classify each prediction
    # --------------------------------------------------------

    ratio = all_detections["inside_ratio_percent"]

    all_detections["classification"] = "FP"
    all_detections["classification_reason"] = "No one-to-one GT match"

    matched_mask = all_detections["has_gt_match"]
    valid_ratio_mask = ratio.notna()

    tp_mask = (
        matched_mask
        & valid_ratio_mask
        & (ratio >= TP_THRESHOLD_PERCENT)
    )

    low_ratio_match_mask = (
        matched_mask
        & valid_ratio_mask
        & (ratio < TP_THRESHOLD_PERCENT)
    )

    invalid_ratio_match_mask = matched_mask & ~valid_ratio_mask

    all_detections.loc[tp_mask, "classification"] = "TP"
    all_detections.loc[
        tp_mask, "classification_reason"
    ] = f"Matched and inside ratio >= {TP_THRESHOLD_PERCENT:.1f}%"

    all_detections.loc[
        low_ratio_match_mask, "classification_reason"
    ] = f"Matched but inside ratio < {TP_THRESHOLD_PERCENT:.1f}%"

    all_detections.loc[
        invalid_ratio_match_mask, "classification_reason"
    ] = "Matched but inside ratio is missing or invalid"

    # --------------------------------------------------------
    # 5. Load GT records for every frame and calculate FN
    # --------------------------------------------------------

    per_frame_rows = []
    gt_details = {}

    frames_to_evaluate = sorted(
        set(EXPECTED_FRAMES)
        | set(predictions["frame_id"].tolist())
        | set(matches["frame_id"].tolist())
    )

    for frame_id in frames_to_evaluate:
        gt_count, gt_records = load_ground_truth_count(frame_id)
        gt_indices = get_gt_indices(gt_records)

        gt_details[frame_id] = {
            "count": gt_count,
            "indices": gt_indices,
        }

        frame_detections = all_detections[
            all_detections["frame_id"] == frame_id
        ]

        frame_matches = matches[matches["frame_id"] == frame_id]

        # TP ground-truth IDs are those assigned to predictions
        # classified as TP, not merely any provisional match.
        frame_tp_rows = frame_detections[
            frame_detections["classification"] == "TP"
        ]

        matched_tp_gt_indices = set(
            frame_tp_rows["matched_gt_index"].dropna().astype(str)
        )

        # Count only IDs that actually exist in this frame's GT file.
        matched_valid_gt_indices = matched_tp_gt_indices & gt_indices

        tp = int((frame_detections["classification"] == "TP").sum())

        # All non-TP predictions count as false positives.
        fp = int((frame_detections["classification"] == "FP").sum())

        # FN = GT records not covered by a true-positive prediction.
        fn = len(gt_indices - matched_valid_gt_indices)

        metrics = calculate_metrics(tp, fp, fn)

        per_frame_rows.append({
            "frame_id": frame_id,
            "ground_truth_count": len(gt_indices),
            "predicted_clouds": len(frame_detections),
            "matched_predictions": int(
                frame_detections["has_gt_match"].sum()
            ),
            "unmatched_predictions": int(
                (~frame_detections["has_gt_match"]).sum()
            ),
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "precision_percent": metrics["precision_percent"],
            "recall_percent": metrics["recall_percent"],
            "f1_percent": metrics["f1_percent"],
        })

    per_frame_df = pd.DataFrame(per_frame_rows)

    # --------------------------------------------------------
    # 6. Calculate overall metrics
    # --------------------------------------------------------

    total_tp = int((all_detections["classification"] == "TP").sum())
    total_fp = int((all_detections["classification"] == "FP").sum())

    all_gt_records = sum(
        details["count"] for details in gt_details.values()
    )

    # Count unique ground-truth indices per frame matched by a TP.
    total_matched_gt = 0

    for frame_id, details in gt_details.items():
        frame_tp_rows = all_detections[
            (all_detections["frame_id"] == frame_id)
            & (all_detections["classification"] == "TP")
        ]

        tp_gt_indices = set(
            frame_tp_rows["matched_gt_index"].dropna().astype(str)
        )

        total_matched_gt += len(tp_gt_indices & details["indices"])

    total_fn = all_gt_records - total_matched_gt

    overall = calculate_metrics(total_tp, total_fp, total_fn)

    overall_df = pd.DataFrame([{
        "frames_evaluated": len(frames_to_evaluate),
        "ground_truth_records": all_gt_records,
        "predicted_clouds": len(all_detections),
        "matched_predictions": int(all_detections["has_gt_match"].sum()),
        "unmatched_predictions": int((~all_detections["has_gt_match"]).sum()),
        "TP": overall["TP"],
        "FP": overall["FP"],
        "FN": overall["FN"],
        "precision_percent": overall["precision_percent"],
        "recall_percent": overall["recall_percent"],
        "f1_percent": overall["f1_percent"],
        "tp_threshold_percent": TP_THRESHOLD_PERCENT,
    }])

    # --------------------------------------------------------
    # 7. Save output CSV files
    # --------------------------------------------------------

    per_frame_file = OUTPUT_DIR / "per_frame_precision_recall_f1.csv"
    overall_file = OUTPUT_DIR / "overall_precision_recall_f1.csv"
    detection_file = OUTPUT_DIR / "per_detection_classification.csv"

    per_frame_df.to_csv(per_frame_file, index=False)
    overall_df.to_csv(overall_file, index=False)
    all_detections.to_csv(detection_file, index=False)

    # --------------------------------------------------------
    # 8. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 72)

    print(f"Frames evaluated:        {len(frames_to_evaluate)}")
    print(f"Ground-truth records:    {all_gt_records}")
    print(f"Predicted clouds:        {len(all_detections)}")
    print(f"Matched predictions:     {int(all_detections['has_gt_match'].sum())}")
    print(f"Unmatched predictions:   {int((~all_detections['has_gt_match']).sum())}")
    print(f"TP:                      {overall['TP']}")
    print(f"FP:                      {overall['FP']}")
    print(f"FN:                      {overall['FN']}")
    print(f"Precision:               {overall['precision_percent']:.2f}%")
    print(f"Recall:                  {overall['recall_percent']:.2f}%")
    print(f"F1-score:                {overall['f1_percent']:.2f}%")

    print("\n" + "=" * 72)
    print("PER-FRAME RESULTS")
    print("=" * 72)

    print(
        per_frame_df[
            [
                "frame_id",
                "ground_truth_count",
                "predicted_clouds",
                "TP",
                "FP",
                "FN",
                "precision_percent",
                "recall_percent",
                "f1_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "precision_percent": "{:.2f}".format,
                "recall_percent": "{:.2f}".format,
                "f1_percent": "{:.2f}".format,
            }
        )
    )

    print("\n" + "=" * 72)
    print("OUTPUT FILES")
    print("=" * 72)
    print(f"Per-frame metrics:       {per_frame_file}")
    print(f"Overall metrics:         {overall_file}")
    print(f"Per-detection results:   {detection_file}")

    print("\nEvaluation completed.")
    print(
        "\nCaution: These metrics are provisional. The GT JSON records do not "
        "contain class labels, the annotation set may include objects other "
        "than cars, and the geometric box interpretation has not been "
        "independently validated. Interpret precision, recall and F1 as "
        "exploratory project metrics, not benchmark detection scores."
    )


if __name__ == "__main__":
    main()