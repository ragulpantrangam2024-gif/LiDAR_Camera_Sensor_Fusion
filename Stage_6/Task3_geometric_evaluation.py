
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 3
# Geometric Evaluation of Matched LiDAR Point Clouds
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MATCHES_CSV = (
    ROOT / "Stage_6" / "results" / "Task1_5"
    / "global_one_to_one_matches.csv"
)

CANDIDATES_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "all_box_candidates.csv"
)

PER_CAR_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "per_car_evaluation.csv"
)

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task3"
PLOTS_DIR = OUTPUT_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_frames(df):
    df = df.copy()
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)
    return df


def numeric_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )
    return df


def save_histogram(values, title, xlabel, output_path):
    values = pd.Series(values).dropna()

    if values.empty:
        print(f"Skipping empty plot: {output_path.name}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=12, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of matches")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 3")
    print("GEOMETRIC EVALUATION")
    print("=" * 72)

    for path in [MATCHES_CSV, CANDIDATES_CSV, PER_CAR_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)

    matches = normalize_frames(pd.read_csv(MATCHES_CSV))
    candidates = normalize_frames(pd.read_csv(CANDIDATES_CSV))
    per_car = normalize_frames(pd.read_csv(PER_CAR_CSV))

    matches = numeric_columns(matches, [
        "points_inside",
        "inside_ratio_percent",
        "center_distance_m",
    ])

    candidates = numeric_columns(candidates, [
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
        "geometry_error_m",
        "finite_points",
    ])

    per_car = numeric_columns(per_car, [
        "input_points",
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
        "geometry_error_m",
    ])

    # Attach point-count and geometric details for the exact
    # cloud-to-GT pair selected by the global matching algorithm.
    detail_columns = [
        "frame_id",
        "car_cloud",
        "gt_index",
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
        "geometry_error_m",
    ]

    match_details = matches.merge(
        candidates[detail_columns].drop_duplicates(
            ["frame_id", "car_cloud", "gt_index"]
        ),
        on=["frame_id", "car_cloud", "gt_index"],
        how="left",
        suffixes=("_match", "_candidate"),
    )

    # Prefer the exact candidate-table measurements.
    for column in [
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
        "geometry_error_m",
    ]:
        candidate_col = f"{column}_candidate"
        match_col = f"{column}_match"

        if candidate_col in match_details.columns:
            match_details[column] = match_details[candidate_col]
        elif match_col in match_details.columns:
            match_details[column] = match_details[match_col]

    # Join each cloud's total finite point count.
    cloud_sizes = per_car[
        ["frame_id", "car_cloud", "finite_points"]
    ].drop_duplicates(["frame_id", "car_cloud"])

    match_details = match_details.merge(
        cloud_sizes,
        on=["frame_id", "car_cloud"],
        how="left",
        suffixes=("", "_cloud"),
    )

    if "finite_points_cloud" in match_details.columns:
        match_details["finite_points"] = (
            match_details["finite_points_cloud"]
            .combine_first(match_details["finite_points"])
        )

    # Derive the outside-point count if absent.
    if "points_outside" not in match_details.columns:
        match_details["points_outside"] = (
            match_details["finite_points"]
            - match_details["points_inside"]
        )

    match_details["outside_ratio_percent"] = (
        match_details["points_outside"]
        .div(match_details["finite_points"].replace(0, np.nan))
        .mul(100)
    )

    match_details["inside_outside_difference"] = (
        match_details["points_inside"]
        - match_details["points_outside"]
    )

    match_details.to_csv(
        OUTPUT_DIR / "per_match_geometric_metrics.csv",
        index=False,
    )

    # Summarize the matched clouds by frame.
    per_frame = (
        match_details.groupby("frame_id")
        .agg(
            matched_clouds=("car_cloud", "nunique"),
            total_finite_points=("finite_points", "sum"),
            total_points_inside=("points_inside", "sum"),
            total_points_outside=("points_outside", "sum"),
            mean_inside_ratio_percent=(
                "inside_ratio_percent", "mean"
            ),
            median_inside_ratio_percent=(
                "inside_ratio_percent", "median"
            ),
            mean_center_distance_m=("center_distance_m", "mean"),
            median_center_distance_m=("center_distance_m", "median"),
        )
        .reset_index()
    )

    per_frame["combined_inside_ratio_percent"] = (
        per_frame["total_points_inside"]
        .div(per_frame["total_finite_points"].replace(0, np.nan))
        .mul(100)
    )

    per_frame.to_csv(
        OUTPUT_DIR / "per_frame_geometric_metrics.csv",
        index=False,
    )

    # Overall descriptive summary.
    overall_rows = []

    metric_definitions = {
        "inside_ratio_percent": "Point containment ratio of each matched cloud",
        "outside_ratio_percent": "Fraction of each matched cloud outside its selected box",
        "center_distance_m": "Distance statistic inherited from the candidate evaluation; not point-to-box distance",
        "geometry_error_m": "Existing Stage 5 geometry diagnostic, if available",
    }

    for column, description in metric_definitions.items():
        if column not in match_details.columns:
            continue

        values = pd.to_numeric(
            match_details[column], errors="coerce"
        ).dropna()

        if values.empty:
            continue

        overall_rows.append({
            "metric": column,
            "count": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "description": description,
        })

    pd.DataFrame(overall_rows).to_csv(
        OUTPUT_DIR / "overall_geometric_metrics.csv",
        index=False,
    )

    # Plots.
    save_histogram(
        match_details["inside_ratio_percent"],
        "Matched Clouds: Point-In-Box Ratio",
        "Inside points (%)",
        PLOTS_DIR / "inside_ratio_distribution.png",
    )

    save_histogram(
        match_details["center_distance_m"],
        "Matched Clouds: Median-Point to Box-Center Distance",
        "Distance (m)",
        PLOTS_DIR / "center_distance_distribution.png",
    )

    if not per_frame.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            per_frame["frame_id"],
            per_frame["combined_inside_ratio_percent"],
        )
        ax.set_title("Combined Point-In-Box Ratio by Frame")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("Inside points (%)")
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / "inside_ratio_by_frame.png",
            dpi=160,
        )
        plt.close(fig)

    print("\n" + "=" * 72)
    print("GEOMETRIC EVALUATION SUMMARY")
    print("=" * 72)

    print("Provisional matches:", len(match_details))
    print(
        "Mean inside ratio (%):",
        round(match_details["inside_ratio_percent"].mean(), 2),
    )
    print(
        "Median inside ratio (%):",
        round(match_details["inside_ratio_percent"].median(), 2),
    )

    print("\nPer-match metric sample:")
    display_columns = [
        "frame_id",
        "car_cloud",
        "gt_index",
        "finite_points",
        "points_inside",
        "points_outside",
        "inside_ratio_percent",
        "center_distance_m",
    ]
    print(
        match_details[
            [c for c in display_columns if c in match_details.columns]
        ].head(10).to_string(index=False)
    )

    print("\nOutput directory:", OUTPUT_DIR)
    print("  per_match_geometric_metrics.csv")
    print("  per_frame_geometric_metrics.csv")
    print("  overall_geometric_metrics.csv")
    print("  plots/")

    print(
        "\nThese measurements describe provisional geometric "
        "associations. They are not 3D IoU, precision, recall, "
        "or validated detection accuracy."
    )


if __name__ == "__main__":
    main()
