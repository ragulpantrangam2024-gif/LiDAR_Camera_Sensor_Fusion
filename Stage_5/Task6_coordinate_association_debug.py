
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"
GT_DIR = ROOT / "bboxes_3D_cam0"
CLOUD_DIR = ROOT / "Stage_3" / "results" / "LiDAR_Mask_Association"

TASK3_CSV = ROOT / "Stage_5" / "results" / "Task3" / "per_car_evaluation.csv"
OUTPUT_DIR = ROOT / "Stage_5" / "results" / "Task6"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    ("0000000100", "car_06_points.npy"),
    ("0000001461", "car_01_points.npy"),
]

# Edges inferred during Task 2B; not yet independently
# verified for every ground-truth box.
EDGE_INDICES = (1, 2, 5)


def load_transform():
    values = np.loadtxt(CALIBRATION_FILE).reshape(-1)

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values, got {values.size}"
        )

    cam_to_lidar = np.eye(4, dtype=np.float64)
    cam_to_lidar[:3, :] = values.reshape(3, 4)

    return cam_to_lidar, np.linalg.inv(cam_to_lidar)


def transform_points(points, matrix):
    homogeneous = np.column_stack(
        [points, np.ones(len(points), dtype=np.float64)]
    )
    return (matrix @ homogeneous.T).T[:, :3]


def load_boxes(frame_id):
    path = GT_DIR / f"BBoxes_{int(frame_id)}.json"

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    boxes = []

    for record in records:
        corners = np.asarray(
            record["corners_cam0"], dtype=np.float64
        )

        if corners.shape == (8, 3) and np.isfinite(corners).all():
            boxes.append({
                "index": record["index"],
                "corners": corners,
                "center": corners.mean(axis=0),
            })

    return boxes


def point_in_box_count(points, corners):
    """
    Test containment using the three edges from corner 0.
    This assumes a parallelepiped and the previously inferred
    corner ordering. Results are diagnostic until that ordering
    is independently verified.
    """
    origin = corners[0]
    edges = corners[list(EDGE_INDICES)] - origin

    try:
        coefficients = np.linalg.solve(
            edges.T, (points - origin).T
        ).T
    except np.linalg.LinAlgError:
        return 0, np.nan

    inside = np.all(
        (coefficients >= -1e-6)
        & (coefficients <= 1.0 + 1e-6),
        axis=1,
    )

    return int(inside.sum()), float(inside.mean() * 100)


def inspect_case(frame_id, cloud_name, cam_to_lidar, lidar_to_cam):
    cloud_path = (
        CLOUD_DIR / frame_id / "car_point_clouds" / cloud_name
    )

    if not cloud_path.exists():
        raise FileNotFoundError(cloud_path)

    cloud = np.load(cloud_path, allow_pickle=False)
    points = np.asarray(cloud[:, :3], dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]

    boxes = load_boxes(frame_id)

    print("\n" + "=" * 72)
    print(f"Frame: {frame_id} | Cloud: {cloud_name}")
    print(f"Finite LiDAR points: {len(points)}")
    print(f"Ground-truth boxes: {len(boxes)}")

    print("\nLiDAR XYZ bounds:")
    print("  Min:", np.round(points.min(axis=0), 3))
    print("  Max:", np.round(points.max(axis=0), 3))
    print("  Median:", np.round(np.median(points, axis=0), 3))

    # Compare both possible transform directions as a diagnostic.
    # Only lidar_to_cam is expected to map LiDAR points into camera
    # coordinates given the current calibration filename convention.
    for transform_name, matrix in [
        ("inverse_calibration (LiDAR -> camera)", lidar_to_cam),
        ("calibration_direct (alternative diagnostic)", cam_to_lidar),
    ]:
        transformed = transform_points(points, matrix)

        print(f"\n--- {transform_name} ---")
        print("XYZ min:", np.round(transformed.min(axis=0), 3))
        print("XYZ max:", np.round(transformed.max(axis=0), 3))
        print("XYZ median:", np.round(np.median(transformed, axis=0), 3))

        ranked = []

        for box in boxes:
            count, ratio = point_in_box_count(
                transformed, box["corners"]
            )
            center_distance = np.linalg.norm(
                np.median(transformed, axis=0) - box["center"]
            )

            ranked.append({
                "gt_index": box["index"],
                "inside_points": count,
                "inside_ratio_percent": ratio,
                "median_to_box_center_m": center_distance,
            })

        ranked.sort(
            key=lambda item: (
                -item["inside_points"],
                item["median_to_box_center_m"],
            )
        )

        print("\nTop 5 candidate boxes:")
        for item in ranked[:5]:
            print(
                f"  GT={item['gt_index']} | "
                f"inside={item['inside_points']} | "
                f"ratio={item['inside_ratio_percent']:.2f}% | "
                f"median-to-center="
                f"{item['median_to_box_center_m']:.2f} m"
            )

    # Report corner edge lengths for the leading candidate
    # from the intended LiDAR-to-camera transform.
    transformed = transform_points(points, lidar_to_cam)
    best_box = max(
        boxes,
        key=lambda box: point_in_box_count(
            transformed, box["corners"]
        )[0],
    )

    corners = best_box["corners"]
    edge_lengths = [
        np.linalg.norm(corners[i] - corners[0])
        for i in EDGE_INDICES
    ]

    print("\nLeading candidate box geometry:")
    print("  GT index:", best_box["index"])
    print("  Center:", np.round(best_box["center"], 3))
    print("  Edge lengths from corner 0 (m):",
          np.round(edge_lengths, 3))


def main():
    print("STAGE 5 - TASK 6.1: COORDINATE DIAGNOSTICS")

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(CALIBRATION_FILE)

    cam_to_lidar, lidar_to_cam = load_transform()

    print("\nCalibration round-trip check:")
    test = np.array([[1.0, 2.0, 10.0]])
    recovered = transform_points(
        transform_points(test, cam_to_lidar), lidar_to_cam
    )
    print("Original:", test[0])
    print("Recovered:", np.round(recovered[0], 8))
    print(
        "Round-trip verifies matrix inversion, not the physical "
        "direction or correctness of the calibration."
    )

    for frame_id, cloud_name in TEST_CASES:
        inspect_case(
            frame_id, cloud_name, cam_to_lidar, lidar_to_cam
        )

    print("\nTask 6.1 diagnostic completed.")
    print("No source data or previous results were modified.")


if __name__ == "__main__":
    main()
