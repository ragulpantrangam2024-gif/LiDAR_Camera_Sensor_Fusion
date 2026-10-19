
from pathlib import Path
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# STAGE 6 - TASK 6
# Ground-Truth 3D Bounding Box Geometry Validation
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GT_ROOT = ROOT / "bboxes_3D_cam0"
OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task6"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_ID = "0000000250"
GT_INDEX = 176


# Hypothesized 12 edges based on the displayed corner coordinates.
# This topology is still provisional and must be validated.
CANDIDATE_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 5), (1, 4), (2, 7), (3, 6),
]


def load_corners(frame_id, gt_index):
    """Load eight ground-truth corners in camera coordinates."""
    path = GT_ROOT / f"BBoxes_{int(frame_id)}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    for record in records:
        if int(record["index"]) == int(gt_index):
            corners = np.asarray(
                record["corners_cam0"], dtype=float
            )

            if corners.shape != (8, 3):
                raise ValueError(
                    f"Expected corners with shape (8, 3), "
                    f"received {corners.shape}"
                )

            if not np.isfinite(corners).all():
                raise ValueError(
                    "Ground-truth corners contain non-finite values."
                )

            return corners

    raise ValueError(
        f"GT index {gt_index} was not found in {path}"
    )


def pairwise_distances(corners):
    """Calculate distances between all pairs of corners."""
    distances = np.zeros((8, 8), dtype=float)

    for i, j in combinations(range(8), 2):
        distance = np.linalg.norm(corners[i] - corners[j])
        distances[i, j] = distance
        distances[j, i] = distance

    return distances


def infer_edges_from_nearest_neighbors(distances):
    """
    Infer provisional edges from mutual nearest neighbors.

    This is a diagnostic only. The result is not assumed to be
    the true box topology.
    """
    nearest = {}

    for i in range(8):
        nearest_indices = np.argsort(distances[i])[1:4]
        nearest[i] = set(nearest_indices.tolist())

    edges = []

    for i, j in combinations(range(8), 2):
        if j in nearest[i] and i in nearest[j]:
            edges.append((i, j))

    return edges, nearest


def edge_report(corners, edges):
    """Return a table of edge endpoints and their lengths."""
    rows = []

    for i, j in edges:
        rows.append({
            "corner_a": i,
            "corner_b": j,
            "length_m": float(
                np.linalg.norm(corners[i] - corners[j])
            ),
        })

    return rows


def plot_box(corners, edges, output_path, title):
    """Plot corners and the supplied candidate edges."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        corners[:, 0],
        corners[:, 1],
        corners[:, 2],
        s=65,
    )

    for i, point in enumerate(corners):
        ax.text(
            point[0],
            point[1],
            point[2],
            str(i),
            fontsize=11,
        )

    for i, j in edges:
        segment = corners[[i, j]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            linewidth=2,
        )

    # Equalize axes for a less distorted view.
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max((maximum - minimum).max() / 2, 0.1)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Y (m)")
    ax.set_zlabel("Camera Z (m)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_edge_lengths(corners, edges, heading):
    """Print and summarize the lengths of candidate edges."""
    print(f"\n{heading}")

    lengths = []

    for i, j in edges:
        length = float(np.linalg.norm(corners[i] - corners[j]))
        lengths.append(length)
        print(f"Corner {i} -- {j}: {length:.4f} m")

    lengths = np.asarray(lengths, dtype=float)

    if len(lengths):
        print("\nEdge length statistics:")
        print(f"Count:   {len(lengths)}")
        print(f"Minimum: {lengths.min():.4f} m")
        print(f"Maximum: {lengths.max():.4f} m")
        print(f"Mean:    {lengths.mean():.4f} m")
        print(f"Std:     {lengths.std():.4f} m")

    return lengths


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 6")
    print("GROUND-TRUTH BOX GEOMETRY VALIDATION")
    print("=" * 72)

    # 1. Load the box.
    corners = load_corners(FRAME_ID, GT_INDEX)
    distances = pairwise_distances(corners)

    print(f"\nFrame: {FRAME_ID}")
    print(f"GT index: {GT_INDEX}")

    # 2. Print corner coordinates.
    print("\nCorner coordinates in camera coordinates:")

    for i, point in enumerate(corners):
        print(
            f"Corner {i}: "
            f"X={point[0]:.4f}, "
            f"Y={point[1]:.4f}, "
            f"Z={point[2]:.4f}"
        )

    # These are coordinate ranges, not actual oriented-box dimensions.
    dimensions = np.ptp(corners, axis=0)

    print("\nAxis-aligned coordinate ranges:")
    print(f"X range: {dimensions[0]:.4f} m")
    print(f"Y range: {dimensions[1]:.4f} m")
    print(f"Z range: {dimensions[2]:.4f} m")

    # 3. Save all 28 pairwise corner distances.
    pair_rows = []

    for i, j in combinations(range(8), 2):
        pair_rows.append({
            "corner_a": i,
            "corner_b": j,
            "distance_m": distances[i, j],
        })

    pair_df = pd.DataFrame(pair_rows)

    pair_df.to_csv(
        OUTPUT_DIR / "all_corner_pair_distances.csv",
        index=False,
    )

    # 4. Test the hypothesized cuboid topology.
    candidate_lengths = print_edge_lengths(
        corners,
        CANDIDATE_EDGES,
        "HYPOTHESIZED CUBOID EDGE LENGTHS",
    )

    candidate_df = pd.DataFrame(
        edge_report(corners, CANDIDATE_EDGES)
    )

    candidate_df.to_csv(
        OUTPUT_DIR / "hypothesized_box_edges.csv",
        index=False,
    )

    # 5. Infer edges independently using nearest neighbors.
    inferred_edges, nearest = infer_edges_from_nearest_neighbors(
        distances
    )

    inferred_df = pd.DataFrame(
        edge_report(corners, inferred_edges),
        columns=["corner_a", "corner_b", "length_m"],
    )

    inferred_df.to_csv(
        OUTPUT_DIR / "inferred_box_edges.csv",
        index=False,
    )

    print("\nNearest corners per corner:")

    for i in range(8):
        neighbors = sorted(
            nearest[i],
            key=lambda j: distances[i, j],
        )

        descriptions = [
            f"{j} ({distances[i, j]:.4f} m)"
            for j in neighbors
        ]

        print(f"Corner {i}: " + ", ".join(descriptions))

    print("\nMutual-nearest-neighbor edges:")

    for i, j in inferred_edges:
        print(
            f"{i} -- {j}: "
            f"{distances[i, j]:.4f} m"
        )

    print(f"\nInferred edge count: {len(inferred_edges)}")
    print("Expected edge count for a cuboid: 12")

    # 6. Save two plots for comparison.
    candidate_plot = (
        OUTPUT_DIR / "gt_box_176_hypothesized_edges.png"
    )

    inferred_plot = (
        OUTPUT_DIR / "gt_box_176_inferred_edges.png"
    )

    plot_box(
        corners,
        CANDIDATE_EDGES,
        candidate_plot,
        f"Frame {FRAME_ID} - GT {GT_INDEX}: "
        "Hypothesized Cuboid Edges",
    )

    plot_box(
        corners,
        inferred_edges,
        inferred_plot,
        f"Frame {FRAME_ID} - GT {GT_INDEX}: "
        "Nearest-Neighbor Edges",
    )

    print("\nGenerated files:")
    print(OUTPUT_DIR / "all_corner_pair_distances.csv")
    print(OUTPUT_DIR / "hypothesized_box_edges.csv")
    print(OUTPUT_DIR / "inferred_box_edges.csv")
    print(candidate_plot)
    print(inferred_plot)

    print("\nIMPORTANT:")
    print(
        "The hypothesized topology is not yet independently "
        "validated against the annotation specification."
    )
    print(
        "Do not use these candidate edges to claim validated "
        "point-in-box metrics until the corner ordering is confirmed."
    )


if __name__ == "__main__":
    main()
