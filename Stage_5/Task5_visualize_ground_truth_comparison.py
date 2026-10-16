
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STAGE 5 - TASK 5
# Visual Evaluation Against Ground-Truth 3D Boxes
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"
GROUND_TRUTH_DIR = ROOT / "bboxes_3D_cam0"

TASK3_DIR = ROOT / "Stage_5" / "results" / "Task3"
TASK4_DIR = ROOT / "Stage_5" / "results" / "Task4"

STAGE3_DIR = (
    ROOT / "Stage_3" / "results" / "LiDAR_Mask_Association"
)

OUTPUT_DIR = ROOT / "Stage_5" / "results" / "Task5"
PLOTS_DIR = OUTPUT_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOW_THRESHOLD = 30.0
HIGH_THRESHOLD = 70.0


# Corner connections for the dataset's expected corner ordering.
BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (0, 5), (1, 4), (2, 7), (3, 6),
    (5, 4), (5, 7), (4, 6), (7, 6),
]


def load_transform(path):
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values; found {values.size}"
        )

    camera_to_lidar = np.eye(4, dtype=np.float64)
    camera_to_lidar[:3, :] = values.reshape(3, 4)

    return np.linalg.inv(camera_to_lidar)


def transform_points(points, transform):
    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )
    return (transform @ homogeneous.T).T[:, :3]


def find_cloud(frame_id, car_cloud):
    path = STAGE3_DIR / frame_id / "car_point_clouds" / car_cloud

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Could not locate cloud: {frame_id}/{car_cloud}"
    )


def find_box(frame_id, gt_index):
    gt_path = GROUND_TRUTH_DIR / f"BBoxes_{int(frame_id)}.json"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    with open(gt_path, "r", encoding="utf-8") as file:
        boxes = json.load(file)

    for obj in boxes:
        if str(obj.get("index")) == str(gt_index):
            corners = np.asarray(
                obj["corners_cam0"], dtype=np.float64
            )

            if corners.shape != (8, 3):
                raise ValueError(
                    f"Unexpected corners shape: {corners.shape}"
                )

            return corners

    raise ValueError(
        f"GT index {gt_index} not found in {gt_path.name}"
    )


def draw_box(ax, corners):
    for start, end in BOX_EDGES:
        a = corners[start]
        b = corners[end]

        ax.plot(
            [a[0], b[0]],
            [a[2], b[2]],
            [a[1], b[1]],
            linewidth=1.5,
        )


def create_plot(row, lidar_to_camera):
    frame_id = str(row["frame_id"]).zfill(10)
    car_cloud = row["car_cloud"]
    gt_index = row["best_gt_index"]

    cloud_path = find_cloud(frame_id, car_cloud)

    cloud = np.load(cloud_path, allow_pickle=False)

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(
            f"Unexpected point cloud shape: {cloud.shape}"
        )

    points_lidar = np.asarray(cloud[:, :3], dtype=np.float64)
    points_lidar = points_lidar[
        np.isfinite(points_lidar).all(axis=1)
    ]

    if len(points_lidar) == 0:
        raise ValueError(f"Empty point cloud: {cloud_path}")

    points_camera = transform_points(
        points_lidar, lidar_to_camera
    )

    corners = find_box(frame_id, gt_index)

    # Camera coordinate convention commonly uses X horizontal,
    # Y vertical, and Z depth. Display Z horizontally in the plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points_camera[:, 0],
        points_camera[:, 2],
        points_camera[:, 1],
        s=2,
        alpha=0.65,
        label="LiDAR car-cloud points",
    )

    draw_box(ax, corners)

    ax.set_xlabel("Camera X (m)")
    ax.set_ylabel("Camera Z / depth (m)")
    ax.set_zlabel("Camera Y (m)")

    ratio = float(row["inside_ratio_percent"])

    ax.set_title(
        f"Frame {frame_id} | {car_cloud} | GT {gt_index}\n"
        f"Best-candidate inside ratio: {ratio:.2f}%"
    )

    ax.legend()
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()

    label = (
        "high_overlap"
        if ratio >= HIGH_THRESHOLD
        else "low_overlap"
    )

    output_path = (
        PLOTS_DIR
        / f"{label}_{frame_id}_{Path(car_cloud).stem}.png"
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    print("=" * 68)
    print("STAGE 5 - TASK 5")
    print("Visual Evaluation Against Ground-Truth 3D Boxes")
    print("=" * 68)

    summary_file = TASK3_DIR / "per_car_evaluation.csv"

    if not summary_file.exists():
        raise FileNotFoundError(summary_file)

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(CALIBRATION_FILE)

    summary = pd.read_csv(summary_file)

    required = {
        "frame_id", "car_cloud",
        "best_gt_index", "inside_ratio_percent",
        "finite_points",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing columns in Task 3 CSV: {sorted(missing)}"
        )

    summary["inside_ratio_percent"] = pd.to_numeric(
        summary["inside_ratio_percent"], errors="coerce"
    )

    summary["finite_points"] = pd.to_numeric(
        summary["finite_points"], errors="coerce"
    )

    eligible = summary[
        summary["inside_ratio_percent"].notna()
        & summary["finite_points"].gt(0)
        & summary["best_gt_index"].notna()
    ].copy()

    high = eligible[
        eligible["inside_ratio_percent"] >= HIGH_THRESHOLD
    ].sort_values(
        "inside_ratio_percent", ascending=False
    )

    low = eligible[
        eligible["inside_ratio_percent"] < LOW_THRESHOLD
    ].sort_values(
        "inside_ratio_percent", ascending=True
    )

    # Select up to three examples from each category.
    selected = pd.concat([
        high.head(3).assign(selection="high_overlap"),
        low.head(3).assign(selection="low_overlap"),
    ])

    selected.to_csv(
        OUTPUT_DIR / "selected_visual_examples.csv",
        index=False,
    )

    if selected.empty:
        print("No suitable examples found.")
        return

    lidar_to_camera = load_transform(CALIBRATION_FILE)

    successful = 0

    for _, row in selected.iterrows():
        try:
            create_plot(row, lidar_to_camera)
            successful += 1
        except Exception as error:
            print(
                f"Could not plot {row['frame_id']} "
                f"{row['car_cloud']}: {error}"
            )

    print("\n" + "=" * 68)
    print("TASK 5 FINISHED")
    print("=" * 68)
    print(f"Examples selected: {len(selected)}")
    print(f"Plots successfully saved: {successful}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Plot directory: {PLOTS_DIR}")
    print(
        "\nThese plots are diagnostic. A selected candidate box "
        "is not necessarily the correct object's ground-truth box."
    )


if __name__ == "__main__":
    main()
