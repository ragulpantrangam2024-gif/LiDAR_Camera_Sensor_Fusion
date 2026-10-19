
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE 6 - TASK 4
# Match Quality Audit
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "Stage_6" / "results" / "Task3"
OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task4"

MATCHES_FILE = INPUT_DIR / "per_match_geometric_metrics.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Review thresholds only; these are NOT ground-truth criteria.
MIN_POINTS_REVIEW = 50
LOW_CONTAINMENT_PERCENT = 40.0
HIGH_CENTER_DISTANCE_M = 10.0


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 4")
    print("MATCH QUALITY AUDIT")
    print("=" * 72)

    if not MATCHES_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file:\n{MATCHES_FILE}\n"
            "Run Task 3 before running Task 4."
        )

    df = pd.read_csv(MATCHES_FILE)

    required_columns = [
        "frame_id",
        "car_cloud",
        "gt_index",
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Normalize identifiers and numeric values.
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)

    numeric_columns = [
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Start with no review flags.
    df["review_flags"] = ""

    def add_flag(mask, flag):
        mask = mask.fillna(False)
        existing = df.loc[mask, "review_flags"]
        df.loc[mask, "review_flags"] = existing.apply(
            lambda value: f"{value}; {flag}".strip("; ")
        )

    add_flag(
        df["finite_points"] < MIN_POINTS_REVIEW,
        f"SMALL_CLOUD (<{MIN_POINTS_REVIEW} points)",
    )

    add_flag(
        df["inside_ratio_percent"] < LOW_CONTAINMENT_PERCENT,
        f"LOW_CONTAINMENT (<{LOW_CONTAINMENT_PERCENT:.0f}%)",
    )

    add_flag(
        df["center_distance_m"] > HIGH_CENTER_DISTANCE_M,
        f"LARGE_CENTER_DISTANCE (>{HIGH_CENTER_DISTANCE_M:.0f} m)",
    )

    add_flag(
        df["inside_ratio_percent"].isna(),
        "MISSING_CONTAINMENT_RATIO",
    )

    add_flag(
        df["center_distance_m"].isna(),
        "MISSING_CENTER_DISTANCE",
    )

    df["needs_review"] = df["review_flags"].ne("")

    # Assign a review category. These are screening labels, not
    # confirmed match correctness labels.
    df["review_category"] = np.select(
        [
            ~df["needs_review"],
            df["review_flags"].str.contains(
                "LARGE_CENTER_DISTANCE", na=False
            ),
            df["review_flags"].str.contains(
                "LOW_CONTAINMENT", na=False
            ),
            df["review_flags"].str.contains(
                "SMALL_CLOUD", na=False
            ),
        ],
        [
            "No threshold triggered",
            "Large center distance",
            "Low containment",
            "Small cloud",
        ],
        default="Multiple or other review flags",
    )

    # Export flagged rows, retaining all available measurements.
    flagged = df[df["needs_review"]].copy()

    flagged = flagged.sort_values(
        by=[
            "center_distance_m",
            "inside_ratio_percent",
            "finite_points",
        ],
        ascending=[False, True, True],
        na_position="last",
    )

    flagged.to_csv(
        OUTPUT_DIR / "flagged_matches.csv",
        index=False,
    )

    # Summary by review category.
    category_summary = (
        df.groupby("review_category", dropna=False)
        .agg(
            matches=("car_cloud", "size"),
            mean_points=("finite_points", "mean"),
            mean_containment_percent=("inside_ratio_percent", "mean"),
            median_containment_percent=("inside_ratio_percent", "median"),
            mean_center_distance_m=("center_distance_m", "mean"),
        )
        .reset_index()
    )

    category_summary.to_csv(
        OUTPUT_DIR / "match_quality_summary.csv",
        index=False,
    )

    # Overall audit statistics.
    total = len(df)
    flagged_count = int(df["needs_review"].sum())
    unflagged_count = total - flagged_count

    multiple_flag_count = int(
        df["review_flags"].str.contains(";", na=False).sum()
    )

    small_cloud_count = int(
        (df["finite_points"] < MIN_POINTS_REVIEW).sum()
    )

    low_containment_count = int(
        (df["inside_ratio_percent"] < LOW_CONTAINMENT_PERCENT).sum()
    )

    large_distance_count = int(
        (df["center_distance_m"] > HIGH_CENTER_DISTANCE_M).sum()
    )

    notes = f"""
STAGE 6 - TASK 4: MATCH QUALITY AUDIT
=====================================

Input:
{MATCHES_FILE}

Output:
{OUTPUT_DIR}

Total provisional matches: {total}
Matches flagged for review: {flagged_count}
Matches without triggered thresholds: {unflagged_count}
Matches with multiple flags: {multiple_flag_count}

Review thresholds:
- Small cloud: fewer than {MIN_POINTS_REVIEW} finite points
- Low containment: inside ratio below {LOW_CONTAINMENT_PERCENT:.1f}%
- Large center distance: center_distance_m above {HIGH_CENTER_DISTANCE_M:.1f} m

Flag counts (categories can overlap):
- Small clouds: {small_cloud_count}
- Low containment: {low_containment_count}
- Large center distance: {large_distance_count}

Interpretation:
These are screening thresholds for prioritizing manual review.
A flagged match is not necessarily incorrect, and an unflagged
match is not necessarily correct.

The containment ratio measures the percentage of points assigned
to a cloud that lie inside its selected ground-truth box.

center_distance_m is the distance statistic supplied by the
existing evaluation pipeline. It must not be described as
point-to-box-surface distance.

Ground-truth annotations do not contain class labels, and the
box geometry and matching assumptions have not been fully
validated. Therefore, this audit does not establish detection
precision, recall, F1, or 3D IoU.
""".strip()

    (OUTPUT_DIR / "review_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    # Print summary.
    print(f"\nTotal provisional matches: {total}")
    print(f"Flagged for review: {flagged_count}")
    print(f"No thresholds triggered: {unflagged_count}")
    print(f"Matches with multiple flags: {multiple_flag_count}")

    print("\nFlag counts (categories may overlap):")
    print(f"  Small clouds: {small_cloud_count}")
    print(f"  Low containment: {low_containment_count}")
    print(f"  Large center distance: {large_distance_count}")

    print("\nFlagged matches:")
    if flagged.empty:
        print("No matches triggered the current review thresholds.")
    else:
        display_columns = [
            "frame_id",
            "car_cloud",
            "gt_index",
            "finite_points",
            "inside_ratio_percent",
            "center_distance_m",
            "review_flags",
        ]

        print(
            flagged[display_columns].to_string(index=False)
        )

    print("\nCategory summary:")
    print(category_summary.to_string(index=False))

    print("\nGenerated files:")
    print(OUTPUT_DIR / "flagged_matches.csv")
    print(OUTPUT_DIR / "match_quality_summary.csv")
    print(OUTPUT_DIR / "review_notes.txt")

    print("\nAudit complete.")
    print("Threshold flags are review prompts, not correctness labels.")


if __name__ == "__main__":
    main()
