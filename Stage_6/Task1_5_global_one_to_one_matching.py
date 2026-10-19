
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# STAGE 6 - TASK 1.5
# Globally Optimal One-to-One Matching
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "all_box_candidates.csv"
)

GREEDY_MATCHES_CSV = (
    ROOT / "Stage_6" / "results" / "Task1"
    / "provisional_one_to_one_matches.csv"
)

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task1_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_INSIDE_RATIO = 30.0
MIN_POINTS_INSIDE = 1

# A large positive reward makes the algorithm prioritize the
# number of eligible matches before optimizing their quality.
MATCH_REWARD = 1_000_000.0
INELIGIBLE_SCORE = -1_000_000_000.0


def normalize_frames(df):
    df = df.copy()
    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)
    return df


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 1.5")
    print("GLOBALLY OPTIMAL ONE-TO-ONE MATCHING")
    print("=" * 72)

    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(CANDIDATES_CSV)

    candidates = normalize_frames(pd.read_csv(CANDIDATES_CSV))

    required = {
        "frame_id",
        "car_cloud",
        "gt_index",
        "points_inside",
        "inside_ratio_percent",
        "center_distance_m",
    }

    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    for col in [
        "points_inside",
        "inside_ratio_percent",
        "center_distance_m",
    ]:
        candidates[col] = pd.to_numeric(
            candidates[col], errors="coerce"
        )

    # Keep candidates meeting the provisional threshold.
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

    matched_records = []

    for frame_id, frame_candidates in candidates.groupby("frame_id"):
        # Every predicted cloud must appear, including clouds that
        # have no candidate meeting the threshold.
        clouds = sorted(frame_candidates["car_cloud"].unique())

        # Only GT boxes with at least one eligible candidate need
        # real columns. Other boxes cannot be matched at this
        # threshold.
        frame_eligible = eligible[
            eligible["frame_id"] == frame_id
        ].copy()

        gt_boxes = sorted(
            frame_eligible["gt_index"].unique(),
            key=str,
        )

        if not clouds or not gt_boxes:
            continue

        cloud_to_row = {cloud: i for i, cloud in enumerate(clouds)}
        gt_to_col = {gt: j for j, gt in enumerate(gt_boxes)}

        # Dummy columns allow each cloud to remain unmatched.
        # Their score is zero; every eligible real match has a
        # large positive reward.
        n_clouds = len(clouds)
        n_gt = len(gt_boxes)

        scores = np.zeros(
            (n_clouds, n_gt + n_clouds),
            dtype=np.float64,
        )

        scores[:, :n_gt] = INELIGIBLE_SCORE

        # Keep one record per cloud/GT pair.
        # Candidate files should already have one row per pair.
        frame_eligible = frame_eligible.drop_duplicates(
            subset=["car_cloud", "gt_index"],
            keep="first",
        )

        for _, row in frame_eligible.iterrows():
            cloud = row["car_cloud"]
            gt = row["gt_index"]

            i = cloud_to_row[cloud]
            j = gt_to_col[gt]

            ratio = float(row["inside_ratio_percent"])
            points_inside = float(row["points_inside"])
            distance = row["center_distance_m"]

            # The reward prioritizes maximum eligible match count.
            # Ratio is the main quality score; point count and
            # distance are small tie-breakers.
            distance_bonus = (
                0.0 if pd.isna(distance)
                else 1.0 / (1.0 + max(float(distance), 0.0))
            )

            quality_score = (
                ratio
                + 0.001 * np.log1p(max(points_inside, 0.0))
                + 0.0001 * distance_bonus
            )

            scores[i, j] = MATCH_REWARD + quality_score

        row_indices, col_indices = linear_sum_assignment(
            scores, maximize=True
        )

        for i, j in zip(row_indices, col_indices):
            # Dummy column means this cloud is unmatched.
            if j >= n_gt:
                continue

            cloud = clouds[i]
            gt = gt_boxes[j]

            selected = frame_eligible[
                (frame_eligible["car_cloud"] == cloud)
                & (frame_eligible["gt_index"] == gt)
            ]

            if selected.empty:
                continue

            row = selected.iloc[0]

            matched_records.append({
                "frame_id": frame_id,
                "car_cloud": cloud,
                "gt_index": gt,
                "points_inside": int(row["points_inside"]),
                "inside_ratio_percent": float(
                    row["inside_ratio_percent"]
                ),
                "center_distance_m": row["center_distance_m"],
                "matching_threshold_percent": MIN_INSIDE_RATIO,
                "match_status": "PROVISIONAL_GLOBAL_MATCH",
            })

    matches = pd.DataFrame(matched_records)

    if matches.empty:
        matches = pd.DataFrame(columns=[
            "frame_id",
            "car_cloud",
            "gt_index",
            "points_inside",
            "inside_ratio_percent",
            "center_distance_m",
            "matching_threshold_percent",
            "match_status",
        ])

    matches.to_csv(
        OUTPUT_DIR / "global_one_to_one_matches.csv",
        index=False,
    )

    # Identify all predicted clouds, including clouds that have
    # no eligible candidate at the chosen threshold.
    all_clouds = (
        candidates[["frame_id", "car_cloud"]]
        .drop_duplicates()
    )

    matched_clouds = (
        matches[["frame_id", "car_cloud"]]
        .drop_duplicates()
    )

    unmatched_clouds = all_clouds.merge(
        matched_clouds,
        on=["frame_id", "car_cloud"],
        how="left",
        indicator=True,
    )

    unmatched_clouds = unmatched_clouds[
        unmatched_clouds["_merge"] == "left_only"
    ].drop(columns="_merge")

    unmatched_clouds["status"] = "UNMATCHED_AT_THRESHOLD"

    unmatched_clouds.to_csv(
        OUTPUT_DIR / "global_unmatched_clouds.csv",
        index=False,
    )

    # Check uniqueness constraints.
    duplicate_clouds = (
        matches.duplicated(["frame_id", "car_cloud"]).sum()
    )
    duplicate_boxes = (
        matches.duplicated(["frame_id", "gt_index"]).sum()
    )

    print("\n" + "=" * 72)
    print("GLOBAL MATCHING SUMMARY")
    print("=" * 72)
    print("Predicted clouds:", len(all_clouds))
    print("Eligible candidate pairs:", len(eligible))
    print("Global one-to-one matches:", len(matches))
    print("Unmatched predicted clouds:", len(unmatched_clouds))
    print("Duplicate cloud assignments:", duplicate_clouds)
    print("Duplicate GT assignments:", duplicate_boxes)
    print("Minimum inside ratio:", f"{MIN_INSIDE_RATIO:.1f}%")

    if not matches.empty:
        print("\nMatched containment statistics:")
        print(
            matches["inside_ratio_percent"]
            .describe()
            .round(2)
            .to_string()
        )

        print("\nGlobal matches by frame:")
        print(matches.groupby("frame_id").size().to_string())

    # Compare with the previous greedy matching result.
    if GREEDY_MATCHES_CSV.exists():
        greedy = normalize_frames(pd.read_csv(GREEDY_MATCHES_CSV))

        print("\n" + "=" * 72)
        print("COMPARISON WITH GREEDY MATCHING")
        print("=" * 72)
        print("Greedy matches:", len(greedy))
        print("Global matches:", len(matches))
        print("Difference:", len(matches) - len(greedy))

        greedy_pairs = set(zip(
            greedy["frame_id"].astype(str),
            greedy["car_cloud"].astype(str),
            greedy["gt_index"].astype(str),
        ))

        global_pairs = set(zip(
            matches["frame_id"].astype(str),
            matches["car_cloud"].astype(str),
            matches["gt_index"].astype(str),
        ))

        print("Identical cloud/GT pairs:", len(greedy_pairs & global_pairs))
        print("Greedy-only pairs:", len(greedy_pairs - global_pairs))
        print("Global-only pairs:", len(global_pairs - greedy_pairs))

    print("\nOutput directory:", OUTPUT_DIR)
    print("  global_one_to_one_matches.csv")
    print("  global_unmatched_clouds.csv")

    print(
        "\nWARNING: These remain provisional geometric matches. "
        "Global assignment prevents duplicate matches but does not "
        "validate object identity, annotation class, calibration, "
        "or box geometry. Do not interpret match count as precision "
        "or recall."
    )


if __name__ == "__main__":
    main()
