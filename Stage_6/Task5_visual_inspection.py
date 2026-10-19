
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# STAGE 6 - TASK 5
# Visual Inspection of Flagged LiDAR-to-GT Matches
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MATCH_FILE = (
    ROOT / "Stage_6" / "results" / "Task3"
    / "per_match_geometric_metrics.csv"
)
CLOUD_ROOT = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)
GT_ROOT = ROOT / "bboxes_3D_cam0"

OUTPUT_DIR = ROOT / "Stage_6" / "results" / "Task5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# First case to inspect.
FRAME_ID = "0000000250"
CAR_CLOUD = "car_02_points.npy"


def load_box_corners(frame_id, gt_index):
    """Load GT box corners, which are stored in camera coordinates."""
    frame_number = int(frame_id)
    gt_file = GT_ROOT / f"BBoxes_{frame_number}.json"

    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    with gt_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        if int(record["index"]) == int(gt_index):
            corners = np.asarray(
                record["corners_cam0"], dtype=float
            )
            if corners.shape != (8, 3):
                raise ValueError(
                    f"Expected (8, 3) corners, got {corners.shape}"
                )
            return corners

    raise ValueError(
        f"GT index {gt_index} not found in {gt_file}"
    )


def load_camera_to_lidar_transform():
    """Read the 3x4 camera-to-LiDAR calibration matrix."""
    calibration_file = ROOT / "calibration" / "calib_cam_to_velo.txt"

    if not calibration_file.exists():
        raise FileNotFoundError(calibration_file)

    values = np.loadtxt(calibration_file).reshape(-1)

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values, got {values.size}"
        )

    T_cam_to_velo = np.eye(4)
    T_cam_to_velo[:3, :] = values.reshape(3, 4)
    return T_cam_to_velo


def transform_camera_to_lidar(points_cam, T_cam_to_velo):
    points_h = np.column_stack(
        [points_cam, np.ones(len(points_cam))]
    )
    return (T_cam_to_velo @ points_h.T).T[:, :3]


def points_inside_box(points, corners):
    """
    Test point inclusion using the project's previously inferred
    box edge convention: corner 0 connects to corners 1, 2 and 5.

    This convention must be validated for the selected annotation.
    """
    edge_vectors = np.vstack([
        corners[1] - corners[0],
        corners[2] - corners[0],
        corners[5] - corners[0],
    ])

    lengths = np.linalg.norm(edge_vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Degenerate bounding-box edge detected")

    unit_axes = edge_vectors / lengths[:, None]
    relative = points - corners[0]

    projections = relative @ unit_axes.T

    inside = np.ones(len(points), dtype=bool)
    for axis in range(3):
        inside &= projections[:, axis] >= -1e-5
        inside &= projections[:, axis] <= lengths[axis] + 1e-5

    return inside


def set_equal_3d_axes(ax, points):
    """Use equal scale on all axes to reduce visual distortion."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max((maximum - minimum).max() / 2, 0.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_box(ax, corners):
    """Draw the 12 box edges using the inferred corner ordering."""
    edges = [
        (0, 1), (0, 2), (0, 5),
        (1, 3), (1, 6),
        (2, 3), (2, 7),
        (5, 6), (5, 7),
        (3, 4), (4, 6), (4, 7),
    ]

    for i, j in edges:
        ax.plot(
            corners[[i, j], 0],
            corners[[i, j], 1],
            corners[[i, j], 2],
            linewidth=2,
        )


def main():
    print("=" * 72)
    print("STAGE 6 - TASK 5")
    print("VISUAL INSPECTION")
    print("=" * 72)

    if not MATCH_FILE.exists():
        raise FileNotFoundError(
            f"Task 3 match file not found: {MATCH_FILE}"
        )

    matches = pd.read_csv(MATCH_FILE)
    matches["frame_id"] = (
        matches["frame_id"].astype(str).str.zfill(10)
    )

    selected = matches[
        (matches["frame_id"] == FRAME_ID)
        & (matches["car_cloud"] == CAR_CLOUD)
    ]

    if selected.empty:
        raise ValueError(
            f"No match found for {FRAME_ID}, {CAR_CLOUD}"
        )

    row = selected.iloc[0]
    gt_index = int(row["gt_index"])

    cloud_file = (
        CLOUD_ROOT / FRAME_ID / "car_point_clouds" / CAR_CLOUD
    )

    if not cloud_file.exists():
        raise FileNotFoundError(
            f"LiDAR cloud not found: {cloud_file}"
        )

    cloud = np.load(cloud_file)

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(
            f"Expected an N x >=3 cloud array, got {cloud.shape}"
        )

    points_lidar = cloud[:, :3]
    finite_mask = np.isfinite(points_lidar).all(axis=1)
    points_lidar = points_lidar[finite_mask]

    if len(points_lidar) == 0:
        raise ValueError("No finite LiDAR points in this cloud")

    corners_cam = load_box_corners(FRAME_ID, gt_index)
    T_cam_to_velo = load_camera_to_lidar_transform()
    corners_lidar = transform_camera_to_lidar(
        corners_cam, T_cam_to_velo
    )

    inside = points_inside_box(points_lidar, corners_lidar)
    outside = ~inside

    print(f"Frame: {FRAME_ID}")
    print(f"Cloud: {CAR_CLOUD}")
    print(f"Selected GT index: {gt_index}")
    print(f"Finite points: {len(points_lidar)}")
    print(f"Points inside: {inside.sum()}")
    print(f"Points outside: {outside.sum()}")
    print(
        "Recomputed containment ratio: "
        f"{100 * inside.mean():.2f}%"
    )
    print(
        "Task 3 recorded containment ratio: "
        f"{float(row['inside_ratio_percent']):.2f}%"
    )
    print(
        "Task 3 center distance (m): "
        f"{float(row['center_distance_m']):.3f}"
    )

    # Save a 3D view in LiDAR coordinates.
    all_geometry = np.vstack([points_lidar, corners_lidar])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    if inside.any():
        ax.scatter(
            points_lidar[inside, 0],
            points_lidar[inside, 1],
            points_lidar[inside, 2],
            s=12,
            label="Inside GT box",
        )

    if outside.any():
        ax.scatter(
            points_lidar[outside, 0],
            points_lidar[outside, 1],
            points_lidar[outside, 2],
            s=12,
            label="Outside GT box",
        )

    plot_box(ax, corners_lidar)

    ax.set_title(
        f"{FRAME_ID} | {CAR_CLOUD}\n"
        f"GT {gt_index} | containment {100 * inside.mean():.1f}%"
    )
    ax.set_xlabel("LiDAR X (m)")
    ax.set_ylabel("LiDAR Y (m)")
    ax.set_zlabel("LiDAR Z (m)")
    ax.legend()
    set_equal_3d_axes(ax, all_geometry)

    fig.tight_layout()
    output_3d = OUTPUT_DIR / "frame_0000000250_car_02_3d.png"
    fig.savefig(output_3d, dpi=180)
    plt.close(fig)

    # Save a top-down view.
    fig, ax = plt.subplots(figsize=(9, 8))

    if inside.any():
        ax.scatter(
            points_lidar[inside, 0],
            points_lidar[inside, 1],
            s=18,
            label="Inside GT box",
        )

    if outside.any():
        ax.scatter(
            points_lidar[outside, 0],
            points_lidar[outside, 1],
            s=18,
            label="Outside GT box",
        )

    closed_box = np.vstack([corners_lidar, corners_lidar[0]])
    ax.plot(
        closed_box[:, 0],
        closed_box[:, 1],
        linewidth=2,
        label="Box corner sequence (reference only)",
    )

    ax.set_title(
        f"Top-down view: {FRAME_ID} | {CAR_CLOUD}"
    )
    ax.set_xlabel("LiDAR X (m)")
    ax.set_ylabel("LiDAR Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_bev = OUTPUT_DIR / "frame_0000000250_car_02_bev.png"
    fig.savefig(output_bev, dpi=180)
    plt.close(fig)

    print("\nSaved visualizations:")
    print(output_3d)
    print(output_bev)
    print("\nImportant:")
    print(
        "The inclusion test relies on an inferred box corner ordering. "
        "Inspect the plots before treating the result as verified."
    )


if __name__ == "__main__":
    main()
