
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE 6 - TASK 1.4
# Provisional One-to-One Ground-Truth Matching
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "all_box_candidates.csv"
)

PER_CAR_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "per_car_evaluation.csv"
)

GT_DIR = ROOT / "bboxes_3D_cam0"

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exploratory threshold, not yet validated.
MIN_INSIDE_RATIO = 30.0
MIN_POINTS_INSIDE = 1


def normalize_frame_ids(df):
    df = df.copy()
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)
    return df


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 1.4")
    print("PROVISIONAL ONE-TO-ONE GROUND-TRUTH MATCHING")
    print("=" * 72)

    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(CANDIDATES_CSV)

    if not PER_CAR_CSV.exists():
        raise FileNotFoundError(PER_CAR_CSV)

    candidates = normalize_frame_ids(pd.read_csv(CANDIDATES_CSV))
    clouds = normalize_frame_ids(pd.read_csv(PER_CAR_CSV))

    required_candidates = {
        "frame_id",
        "car_cloud",
        "gt_index",
        "points_inside",
        "inside_ratio_percent",
        "center_distance_m",
    }

    missing = required_candidates - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    required_clouds = {"frame_id", "car_cloud"}
    missing = required_clouds - set(clouds.columns)
    if missing:
        raise ValueError(
            f"Missing per-cloud columns: {sorted(missing)}"
        )

    for col in [
        "points_inside",
        "inside_ratio_percent",
        "center_distance_m",
    ]:
        candidates[col] = pd.to_numeric(
            candidates[col], errors="coerce"
        )

    # Keep candidates that meet the provisional threshold.
    eligible = candidates[
        candidates["points_inside"].ge(MIN_POINTS_INSIDE)
        & candidates["inside_ratio_percent"].ge(MIN_INSIDE_RATIO)
    ].copy()

    eligible = eligible.dropna(
        subset=[
            "frame_id",
            "car_cloud",
            "gt_index",
            "points_inside",
            "inside_ratio_percent",
        ]
    )

    # A greedy global ranking within each frame:
    # 1. Higher inside ratio
    # 2. More points inside
    # 3. Smaller center distance as a tie-breaker
    eligible = eligible.sort_values(
        [
            "frame_id",
            "inside_ratio_percent",
            "points_inside",
            "center_distance_m",
        ],
        ascending=[True, False, False, True],
        na_position="last",
    )

    matched_rows = []
    used_clouds = set()
    used_boxes = set()

    # Process frames independently.
    for frame_id, frame_candidates in eligible.groupby("frame_id"):
        frame_used_clouds = set()
        frame_used_boxes = set()

        for _, row in frame_candidates.iterrows():
            cloud_key = str(row["car_cloud"])
            gt_key = str(row["gt_index"])

            if cloud_key in frame_used_clouds:
                continue

            if gt_key in frame_used_boxes:
                continue

            matched_rows.append({
                "frame_id": frame_id,
                "car_cloud": cloud_key,
                "gt_index": gt_key,
                "points_inside": int(row["points_inside"]),
                "inside_ratio_percent": float(
                    row["inside_ratio_percent"]
                ),
                "center_distance_m": row["center_distance_m"],
                "match_status": "PROVISIONAL_MATCH",
                "matching_threshold_percent": MIN_INSIDE_RATIO,
            })

            frame_used_clouds.add(cloud_key)
            frame_used_boxes.add(gt_key)

        used_clouds.update(
            (frame_id, cloud) for cloud in frame_used_clouds
        )
        used_boxes.update(
            (frame_id, box) for box in frame_used_boxes
        )

    matches = pd.DataFrame(matched_rows)

    if matches.empty:
        matches = pd.DataFrame(columns=[
            "frame_id",
            "car_cloud",
            "gt_index",
            "points_inside",
            "inside_ratio_percent",
            "center_distance_m",
            "match_status",
            "matching_threshold_percent",
        ])

    matches.to_csv(
        OUTPUT_DIR / "provisional_one_to_one_matches.csv",
        index=False,
    )

    # Export unmatched predicted clouds.
    cloud_keys = clouds[["frame_id", "car_cloud"]].drop_duplicates()
    matched_cloud_keys = matches[["frame_id", "car_cloud"]].drop_duplicates()

    unmatched_clouds = cloud_keys.merge(
        matched_cloud_keys,
        on=["frame_id", "car_cloud"],
        how="left",
        indicator=True,
    )

    unmatched_clouds = unmatched_clouds[
        unmatched_clouds["_merge"] == "left_only"
    ].drop(columns="_merge")

    unmatched_clouds["status"] = "UNMATCHED_CLOUD"
    unmatched_clouds.to_csv(
        OUTPUT_DIR / "unmatched_predicted_clouds.csv",
        index=False,
    )

    # Load annotation record counts for every available frame,
    # including frames with zero predicted clouds.
    gt_rows = []

    for path in sorted(GT_DIR.glob("BBoxes_*.json")):
        try:
            frame_number = path.stem.replace("BBoxes_", "")
            frame_id = frame_number.zfill(10)

            import json
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)

            gt_rows.append({
                "frame_id": frame_id,
                "ground_truth_records": len(records),
            })
        except Exception as error:
            print(f"Could not read {path.name}: {error}")

    gt_counts = pd.DataFrame(
        gt_rows,
        columns=["frame_id", "ground_truth_records"],
    )

    if not gt_counts.empty:
        matched_counts = (
            matches.groupby("frame_id")["gt_index"]
            .nunique()
            .rename("provisional_matched_boxes")
            .reset_index()
        )

        gt_counts = gt_counts.merge(
            matched_counts,
            on="frame_id",
            how="left",
        )

        gt_counts["provisional_matched_boxes"] = (
            gt_counts["provisional_matched_boxes"]
            .fillna(0).astype(int)
        )

        gt_counts["unmatched_gt_records"] = (
            gt_counts["ground_truth_records"]
            - gt_counts["provisional_matched_boxes"]
        )

        gt_counts.to_csv(
            OUTPUT_DIR / "ground_truth_matching_summary.csv",
            index=False,
        )

    print("\n" + "=" * 72)
    print("MATCHING SUMMARY")
    print("=" * 72)
    print(f"Input predicted clouds: {len(cloud_keys)}")
    print(f"Eligible candidate pairs: {len(eligible)}")
    print(f"Provisional one-to-one matches: {len(matches)}")
    print(f"Unmatched predicted clouds: {len(unmatched_clouds)}")
    print(f"Minimum inside ratio: {MIN_INSIDE_RATIO:.1f}%")

    if not matches.empty:
        print("\nMatched containment statistics:")
        print(
            matches["inside_ratio_percent"]
            .describe()
            .round(2)
            .to_string()
        )

        print("\nMatches by frame:")
        print(
            matches.groupby("frame_id")
            .size()
            .to_string()
        )

    print("\nOutput directory:", OUTPUT_DIR)
    print("\nGenerated files:")
    print("  provisional_one_to_one_matches.csv")
    print("  unmatched_predicted_clouds.csv")
    print("  ground_truth_matching_summary.csv")
    print(
        "\nWARNING: These are provisional geometric matches. "
        "They are not confirmed true positives, and the unmatched "
        "GT records cannot automatically be interpreted as false "
        "negatives because annotation classes and box geometry "
        "have not been fully validated."
    )


if __name__ == "__main__":
    main()
