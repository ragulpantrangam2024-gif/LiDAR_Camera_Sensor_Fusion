
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STAGE 5 - TASK 4
# Quantitative Fusion Evaluation and Metrics
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TASK3_DIR = ROOT / "Stage_5" / "results" / "Task3"
INPUT_SUMMARY = TASK3_DIR / "per_car_evaluation.csv"
INPUT_CANDIDATES = TASK3_DIR / "all_box_candidates.csv"

OUTPUT_DIR = ROOT / "Stage_5" / "results" / "Task4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_FILE = OUTPUT_DIR / "overall_metrics.csv"
FRAME_FILE = OUTPUT_DIR / "per_frame_metrics.csv"
QUALITY_FILE = OUTPUT_DIR / "cloud_quality_analysis.csv"
LOW_OVERLAP_FILE = OUTPUT_DIR / "low_overlap_clouds.csv"

LOW_OVERLAP_THRESHOLD = 30.0
HIGH_OVERLAP_THRESHOLD = 70.0
SPARSE_POINT_THRESHOLD = 20


def main():
    print("=" * 68)
    print("STAGE 5 - TASK 4")
    print("Quantitative Fusion Evaluation and Metrics")
    print("=" * 68)

    # --------------------------------------------------------
    # 1. Load Task 3 results
    # --------------------------------------------------------

    if not INPUT_SUMMARY.exists():
        raise FileNotFoundError(
            f"Task 3 summary not found: {INPUT_SUMMARY}"
        )

    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Task 3 candidates file not found: {INPUT_CANDIDATES}"
        )

    summary = pd.read_csv(INPUT_SUMMARY)
    candidates = pd.read_csv(INPUT_CANDIDATES)

    if summary.empty:
        raise ValueError("Task 3 summary contains no records.")

    required_summary = {
        "frame_id",
        "car_cloud",
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "status",
    }

    missing = required_summary - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing columns in Task 3 summary: {sorted(missing)}"
        )

    summary["inside_ratio_percent"] = pd.to_numeric(
        summary["inside_ratio_percent"], errors="coerce"
    )

    summary["finite_points"] = pd.to_numeric(
        summary["finite_points"], errors="coerce"
    )

    summary["points_inside"] = pd.to_numeric(
        summary["points_inside"], errors="coerce"
    )

    summary["points_outside"] = pd.to_numeric(
        summary["points_outside"], errors="coerce"
    )

    # Keep clouds with a valid computed ratio for ratio metrics.
    valid = summary[
        summary["inside_ratio_percent"].notna()
        & summary["finite_points"].gt(0)
    ].copy()

    if valid.empty:
        raise ValueError("No non-empty clouds have valid ratios.")

    # --------------------------------------------------------
    # 2. Overall diagnostic metrics
    # --------------------------------------------------------

    ratios = valid["inside_ratio_percent"]

    total_clouds = len(summary)
    evaluated_clouds = len(valid)

    any_inside = valid["points_inside"] > 0
    low_overlap = ratios < LOW_OVERLAP_THRESHOLD
    high_overlap = ratios >= HIGH_OVERLAP_THRESHOLD

    total_finite_points = int(
        summary["finite_points"].fillna(0).sum()
    )

    total_inside_best = int(
        summary["points_inside"].fillna(0).sum()
    )

    total_outside_best = int(
        summary["points_outside"].fillna(0).sum()
    )

    overall_rows = [
        ("car_cloud_records", total_clouds),
        ("non_empty_clouds_with_valid_ratio", evaluated_clouds),
        ("empty_clouds", int((summary["status"] == "EMPTY_CLOUD").sum())),
        (
            "sparse_clouds",
            int((summary["finite_points"] < SPARSE_POINT_THRESHOLD).sum()),
        ),
        ("total_finite_points_across_clouds", total_finite_points),
        ("inside_points_for_selected_best_candidates", total_inside_best),
        ("outside_points_for_selected_best_candidates", total_outside_best),
        ("mean_best_candidate_inside_ratio_percent", round(ratios.mean(), 3)),
        ("median_best_candidate_inside_ratio_percent", round(ratios.median(), 3)),
        ("minimum_inside_ratio_percent", round(ratios.min(), 3)),
        ("maximum_inside_ratio_percent", round(ratios.max(), 3)),
        ("clouds_with_at_least_one_point_inside", int(any_inside.sum())),
        ("clouds_with_zero_points_inside", int((~any_inside).sum())),
        (
            "clouds_below_30_percent_inside",
            int(low_overlap.sum()),
        ),
        (
            "clouds_at_least_70_percent_inside",
            int(high_overlap.sum()),
        ),
        (
            "percent_clouds_below_30_percent_inside",
            round(100 * low_overlap.mean(), 3),
        ),
        (
            "percent_clouds_at_least_70_percent_inside",
            round(100 * high_overlap.mean(), 3),
        ),
    ]

    overall = pd.DataFrame(
        overall_rows, columns=["metric", "value"]
    )

    overall.to_csv(OVERALL_FILE, index=False)

    # --------------------------------------------------------
    # 3. Per-frame metrics
    # --------------------------------------------------------

    frame_rows = []

    for frame_id, group in summary.groupby("frame_id"):
        non_empty = group[
            group["finite_points"].fillna(0) > 0
        ].copy()

        valid_frame = non_empty[
            non_empty["inside_ratio_percent"].notna()
        ]

        frame_ratios = valid_frame["inside_ratio_percent"]

        frame_rows.append({
            "frame_id": frame_id,
            "car_cloud_records": len(group),
            "non_empty_clouds": len(non_empty),
            "empty_clouds": int(
                (group["status"] == "EMPTY_CLOUD").sum()
            ),
            "sparse_clouds": int(
                (group["finite_points"] < SPARSE_POINT_THRESHOLD).sum()
            ),
            "clouds_with_any_points_inside": int(
                (valid_frame["points_inside"] > 0).sum()
            ),
            "mean_inside_ratio_percent": (
                round(frame_ratios.mean(), 3)
                if len(frame_ratios) else np.nan
            ),
            "median_inside_ratio_percent": (
                round(frame_ratios.median(), 3)
                if len(frame_ratios) else np.nan
            ),
            "clouds_below_30_percent": int(
                (frame_ratios < LOW_OVERLAP_THRESHOLD).sum()
            ),
            "clouds_at_least_70_percent": int(
                (frame_ratios >= HIGH_OVERLAP_THRESHOLD).sum()
            ),
        })

    per_frame = pd.DataFrame(frame_rows)
    per_frame.to_csv(FRAME_FILE, index=False)

    # --------------------------------------------------------
    # 4. Cloud quality categories
    # --------------------------------------------------------

    quality = summary.copy()

    def classify(row):
        points = row["finite_points"]
        ratio = row["inside_ratio_percent"]

        if pd.isna(points) or points == 0:
            return "EMPTY_OR_UNAVAILABLE"

        if points < SPARSE_POINT_THRESHOLD:
            return "SPARSE_CLOUD"

        if pd.isna(ratio):
            return "RATIO_UNAVAILABLE"

        if ratio < LOW_OVERLAP_THRESHOLD:
            return "LOW_OVERLAP"

        if ratio < HIGH_OVERLAP_THRESHOLD:
            return "MODERATE_OVERLAP"

        return "HIGH_OVERLAP"

    quality["quality_category"] = quality.apply(classify, axis=1)

    quality.to_csv(QUALITY_FILE, index=False)

    # --------------------------------------------------------
    # 5. Low-overlap cases for further inspection
    # --------------------------------------------------------

    low_cases = quality[
        quality["quality_category"].isin([
            "LOW_OVERLAP",
            "SPARSE_CLOUD",
            "EMPTY_OR_UNAVAILABLE",
        ])
    ].copy()

    low_cases.to_csv(LOW_OVERLAP_FILE, index=False)

    # --------------------------------------------------------
    # 6. Compare best and second-best candidate boxes
    # --------------------------------------------------------

    if not candidates.empty:
        needed = {
            "frame_id",
            "car_cloud",
            "gt_index",
            "inside_ratio_percent",
        }

        if needed.issubset(candidates.columns):
            candidates["inside_ratio_percent"] = pd.to_numeric(
                candidates["inside_ratio_percent"], errors="coerce"
            )

            candidates = candidates.dropna(
                subset=["inside_ratio_percent"]
            )

            candidates = candidates.sort_values(
                "inside_ratio_percent",
                ascending=False,
            )

            ranked = candidates.groupby(
                ["frame_id", "car_cloud"], as_index=False
            ).head(2).copy()

            ranked["candidate_rank"] = ranked.groupby(
                ["frame_id", "car_cloud"]
            )["inside_ratio_percent"].rank(
                method="first", ascending=False
            )

            top_two = ranked.pivot(
                index=["frame_id", "car_cloud"],
                columns="candidate_rank",
                values="inside_ratio_percent",
            ).reset_index()

            top_two.columns.name = None
            top_two = top_two.rename(columns={
                1.0: "best_candidate_ratio_percent",
                2.0: "second_candidate_ratio_percent",
            })

            if "second_candidate_ratio_percent" not in top_two.columns:
                top_two["second_candidate_ratio_percent"] = np.nan

            top_two["candidate_ratio_gap_percent_points"] = (
                top_two["best_candidate_ratio_percent"]
                - top_two["second_candidate_ratio_percent"]
            )

            quality = quality.merge(
                top_two,
                on=["frame_id", "car_cloud"],
                how="left",
            )

            quality.to_csv(QUALITY_FILE, index=False)

    # --------------------------------------------------------
    # 7. Print results
    # --------------------------------------------------------

    print("\nOVERALL DIAGNOSTIC METRICS")
    print("-" * 68)

    for _, row in overall.iterrows():
        print(f"{row['metric']}: {row['value']}")

    print("\nQUALITY CATEGORY COUNTS")
    print("-" * 68)
    print(quality["quality_category"].value_counts().to_string())

    print("\nOUTPUT FILES")
    print("-" * 68)
    print(f"Overall metrics: {OVERALL_FILE}")
    print(f"Per-frame metrics: {FRAME_FILE}")
    print(f"Cloud quality analysis: {QUALITY_FILE}")
    print(f"Low-overlap clouds: {LOW_OVERLAP_FILE}")

    print("\nTASK 4 COMPLETED")
    print(
        "These metrics describe point containment in the selected "
        "candidate boxes. They are not detection precision, recall, "
        "or 3D IoU accuracy."
    )


if __name__ == "__main__":
    main()
