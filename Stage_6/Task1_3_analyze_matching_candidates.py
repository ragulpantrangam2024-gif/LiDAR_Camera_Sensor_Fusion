
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "Stage_5" / "results" / "Task3"
    / "all_box_candidates.csv"
)

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 1.3")
    print("ONE-TO-ONE MATCHING CANDIDATE ANALYSIS")
    print("=" * 72)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required = {
        "frame_id", "car_cloud", "gt_index",
        "finite_points", "points_inside",
        "inside_ratio_percent", "center_distance_m",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["frame_id"] = df["frame_id"].astype(str).str.zfill(10)

    for col in [
        "points_inside", "inside_ratio_percent",
        "center_distance_m", "finite_points",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cloud_key = ["frame_id", "car_cloud"]

    # Rank candidate boxes for each cloud.
    # A box with more points inside is preferred; ratio and
    # center distance break ties.
    ranked = df.sort_values(
        cloud_key + [
            "points_inside",
            "inside_ratio_percent",
            "center_distance_m",
        ],
        ascending=[True, True, False, False, True],
        na_position="last",
    ).copy()

    ranked["candidate_rank"] = (
        ranked.groupby(cloud_key).cumcount() + 1
    )

    best = ranked[ranked["candidate_rank"] == 1].copy()

    best.to_csv(
        OUTPUT_DIR / "best_candidate_per_cloud.csv",
        index=False,
    )

    print("\nClouds:", len(best))
    print("Clouds with at least one point inside the best box:",
          int((best["points_inside"] > 0).sum()))
    print("Clouds with zero points inside the best box:",
          int((best["points_inside"] == 0).sum()))

    print("\nBest-candidate containment distribution:")
    print(
        best["inside_ratio_percent"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(2)
        .to_string()
    )

    print("\nCandidate ranking diagnostics:")
    for threshold in [1, 5, 10, 20, 30, 50, 70]:
        count = int(
            (best["inside_ratio_percent"] >= threshold).sum()
        )
        print(
            f"Clouds with best-candidate ratio >= {threshold:>2}%: "
            f"{count}/{len(best)}"
        )

    # Provisional eligibility counts only. These are not final
    # true positives because box identity and geometry are not
    # yet independently validated.
    print("\nProvisional candidate eligibility counts:")
    for threshold in [1, 10, 30, 50, 70]:
        eligible = df[
            (df["points_inside"] > 0)
            & (df["inside_ratio_percent"] >= threshold)
        ]
        print(
            f"Candidate pairs with ratio >= {threshold:>2}% "
            f"and >=1 point inside: {len(eligible)}"
        )

    # Save all candidates ranked within each cloud so we can
    # inspect alternatives before enforcing one-to-one matching.
    ranked.to_csv(
        OUTPUT_DIR / "ranked_candidates_per_cloud.csv",
        index=False,
    )

    # Candidate multiplicity summary: how many distinct clouds
    # have a particular GT box among their top-ranked candidates.
    for top_n in [1, 3, 5]:
        top = ranked[ranked["candidate_rank"] <= top_n]
        counts = (
            top.groupby(["frame_id", "gt_index"])["car_cloud"]
            .nunique()
            .reset_index(name="distinct_clouds")
        )
        repeated = counts[counts["distinct_clouds"] > 1]
        print(
            f"\nFrame/GT pairs shared by multiple clouds "
            f"within top {top_n}: {len(repeated)}"
        )

    print("\nFiles saved to:", OUTPUT_DIR)
    print(
        "\nNo final matches or detection metrics have been declared. "
        "Thresholds above are exploratory, not validated."
    )


if __name__ == "__main__":
    main()
