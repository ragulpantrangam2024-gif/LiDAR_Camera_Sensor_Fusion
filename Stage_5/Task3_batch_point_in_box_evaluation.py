
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# STAGE 5 - TASK 3
# Batch Point-in-Box Evaluation Across All Frames
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"
GROUND_TRUTH_DIR = ROOT / "bboxes_3D_cam0"
STAGE3_DIR = ROOT / "Stage_3" / "results"

OUTPUT_DIR = ROOT / "Stage_5" / "results" / "Task3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "per_car_evaluation.csv"
CANDIDATES_FILE = OUTPUT_DIR / "all_box_candidates.csv"
WARNINGS_FILE = OUTPUT_DIR / "evaluation_warnings.csv"

SPARSE_THRESHOLD = 20
GEOMETRY_TOLERANCE_M = 0.10
POINT_TOLERANCE = 1e-6


# ============================================================
# CALIBRATION
# ============================================================

def load_transform(path):
    """Load the 3x4 camera-to-LiDAR calibration matrix."""
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values; found {values.size}"
        )

    camera_to_lidar = np.eye(4, dtype=np.float64)
    camera_to_lidar[:3, :] = values.reshape(3, 4)

    return camera_to_lidar


def transform_points(points, transform):
    points_h = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )
    return (transform @ points_h.T).T[:, :3]


# ============================================================
# FILE DISCOVERY
# ============================================================


def discover_clouds():
    canonical_dir = (
        STAGE3_DIR / "LiDAR_Mask_Association"
    )

    if not canonical_dir.exists():
        raise FileNotFoundError(
            f"Canonical association directory not found: "
            f"{canonical_dir}"
        )

    return sorted(
        canonical_dir.glob(
            "*/car_point_clouds/car_*_points.npy"
        )
    )

def get_frame_id(path):
    for parent in path.parents:
        if re.fullmatch(r"\d{10}", parent.name):
            return parent.name

    return None


def find_ground_truth(frame_id):
    expected = GROUND_TRUTH_DIR / f"BBoxes_{int(frame_id)}.json"

    if expected.exists():
        return expected

    for path in GROUND_TRUTH_DIR.glob("BBoxes_*.json"):
        match = re.fullmatch(r"BBoxes_(\d+)\.json", path.name)

        if match and int(match.group(1)) == int(frame_id):
            return path

    return None


# ============================================================
# GROUND-TRUTH BOX GEOMETRY
# ============================================================

def load_boxes(path):
    with open(path, "r", encoding="utf-8") as file:
        objects = json.load(file)

    if not isinstance(objects, list):
        raise ValueError("Expected a list of ground-truth boxes.")

    boxes = []

    for obj in objects:
        if not isinstance(obj, dict) or "corners_cam0" not in obj:
            continue

        corners = np.asarray(obj["corners_cam0"], dtype=np.float64)

        if corners.shape != (8, 3):
            continue

        if not np.isfinite(corners).all():
            continue

        boxes.append({
            "index": obj.get("index", "unknown"),
            "corners": corners,
        })

    return boxes


def get_box_basis(corners):
    """
    Use edges 0->1, 0->2 and 0->5.

    This corner ordering must be validated for the dataset.
    Reject boxes that do not match the expected parallelepiped.
    """
    origin = corners[0]

    edge1 = corners[1] - origin
    edge2 = corners[2] - origin
    edge3 = corners[5] - origin

    basis = np.column_stack((edge1, edge2, edge3))

    if abs(np.linalg.det(basis)) < 1e-8:
        raise ValueError("Degenerate box basis.")

    # Expected coordinates in the dataset's corner ordering.
    expected = np.array([
        origin,
        origin + edge1,
        origin + edge2,
        origin + edge1 + edge2,
        origin + edge3,
        origin + edge1 + edge3,
        origin + edge2 + edge3,
        origin + edge1 + edge2 + edge3,
    ])

    actual = corners[[0, 1, 2, 3, 5, 4, 7, 6]]

    errors = np.linalg.norm(actual - expected, axis=1)
    max_error = float(errors.max())

    if max_error > GEOMETRY_TOLERANCE_M:
        raise ValueError(
            f"Corner-order/geometry mismatch: {max_error:.3f} m"
        )

    return origin, basis, max_error


def points_inside_box(points_camera, corners):
    origin, basis, geometry_error = get_box_basis(corners)

    local_coordinates = np.linalg.solve(
        basis, (points_camera - origin).T
    ).T

    mask = np.all(
        (local_coordinates >= -POINT_TOLERANCE)
        & (local_coordinates <= 1.0 + POINT_TOLERANCE),
        axis=1,
    )

    return mask, geometry_error


# ============================================================
# EVALUATION
# ============================================================

def main():
    print("=" * 68)
    print("STAGE 5 - TASK 3")
    print("Batch Point-in-Box Evaluation Across All Frames")
    print("=" * 68)

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(CALIBRATION_FILE)

    if not GROUND_TRUTH_DIR.exists():
        raise FileNotFoundError(GROUND_TRUTH_DIR)

    if not STAGE3_DIR.exists():
        raise FileNotFoundError(STAGE3_DIR)

    camera_to_lidar = load_transform(CALIBRATION_FILE)
    lidar_to_camera = np.linalg.inv(camera_to_lidar)

    cloud_files = discover_clouds()

    if not cloud_files:
        raise FileNotFoundError(
            "No car point-cloud files were found under Stage_3/results."
        )

    print(f"Cloud files discovered: {len(cloud_files)}")

    summary_rows = []
    candidate_rows = []
    warning_rows = []

    # Cache boxes so each frame's JSON is loaded only once.
    box_cache = {}

    for position, cloud_path in enumerate(cloud_files, start=1):
        frame_id = get_frame_id(cloud_path)

        print(
            f"\n[{position}/{len(cloud_files)}] "
            f"{frame_id or 'UNKNOWN FRAME'} / {cloud_path.name}"
        )

        if frame_id is None:
            warning_rows.append({
                "frame_id": "",
                "car_cloud": cloud_path.name,
                "warning": "Frame ID not found in parent directories",
                "details": str(cloud_path),
            })
            continue

        if frame_id not in box_cache:
            gt_path = find_ground_truth(frame_id)

            if gt_path is None:
                box_cache[frame_id] = None
            else:
                try:
                    box_cache[frame_id] = {
                        "path": gt_path,
                        "boxes": load_boxes(gt_path),
                    }
                except Exception as error:
                    box_cache[frame_id] = None
                    warning_rows.append({
                        "frame_id": frame_id,
                        "car_cloud": "",
                        "warning": "Could not read ground-truth file",
                        "details": str(error),
                    })

        frame_gt = box_cache[frame_id]

        if frame_gt is None:
            warning_rows.append({
                "frame_id": frame_id,
                "car_cloud": cloud_path.name,
                "warning": "Ground-truth file unavailable",
                "details": "",
            })
            continue

        try:
            cloud = np.load(cloud_path, allow_pickle=False)
        except Exception as error:
            warning_rows.append({
                "frame_id": frame_id,
                "car_cloud": cloud_path.name,
                "warning": "Could not load point cloud",
                "details": str(error),
            })
            continue

        if cloud.ndim != 2 or cloud.shape[1] < 3:
            warning_rows.append({
                "frame_id": frame_id,
                "car_cloud": cloud_path.name,
                "warning": "Unexpected point-cloud shape",
                "details": str(cloud.shape),
            })
            continue

        xyz = np.asarray(cloud[:, :3], dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        number_of_points = len(xyz)

        base = {
            "frame_id": frame_id,
            "car_cloud": cloud_path.name,
            "ground_truth_file": frame_gt["path"].name,
            "input_points": len(cloud),
            "finite_points": number_of_points,
            "sparse_cloud": number_of_points < SPARSE_THRESHOLD,
        }

        if number_of_points == 0:
            summary_rows.append({
                **base,
                "valid_boxes": 0,
                "best_gt_index": "",
                "points_inside": 0,
                "points_outside": 0,
                "inside_ratio_percent": 0.0,
                "center_distance_m": "",
                "geometry_error_m": "",
                "status": "EMPTY_CLOUD",
            })
            continue

        points_camera = transform_points(xyz, lidar_to_camera)
        median_point = np.median(points_camera, axis=0)

        candidates = []
        valid_box_count = 0

        for box in frame_gt["boxes"]:
            try:
                inside_mask, geometry_error = points_inside_box(
                    points_camera, box["corners"]
                )
            except Exception as error:
                warning_rows.append({
                    "frame_id": frame_id,
                    "car_cloud": cloud_path.name,
                    "warning": "Ground-truth box rejected",
                    "details": f"GT {box['index']}: {error}",
                })
                continue

            valid_box_count += 1
            inside_count = int(inside_mask.sum())
            outside_count = number_of_points - inside_count
            ratio = 100.0 * inside_count / number_of_points

            center = box["corners"].mean(axis=0)
            center_distance = float(np.linalg.norm(median_point - center))

            row = {
                "frame_id": frame_id,
                "car_cloud": cloud_path.name,
                "gt_index": box["index"],
                "finite_points": number_of_points,
                "points_inside": inside_count,
                "points_outside": outside_count,
                "inside_ratio_percent": round(ratio, 3),
                "center_distance_m": round(center_distance, 3),
                "geometry_error_m": round(geometry_error, 6),
            }

            candidates.append(row)
            candidate_rows.append(row)

        if not candidates:
            summary_rows.append({
                **base,
                "valid_boxes": 0,
                "best_gt_index": "",
                "points_inside": 0,
                "points_outside": number_of_points,
                "inside_ratio_percent": 0.0,
                "center_distance_m": "",
                "geometry_error_m": "",
                "status": "NO_VALID_BOXES",
            })
            continue

        # Candidate ranking is diagnostic, not a confirmed object match.
        candidates.sort(
            key=lambda row: (
                row["points_inside"],
                row["inside_ratio_percent"],
                -row["center_distance_m"],
            ),
            reverse=True,
        )

        best = candidates[0]

        status = (
            "SPARSE_CLOUD"
            if number_of_points < SPARSE_THRESHOLD
            else "OK"
        )

        summary_rows.append({
            **base,
            "valid_boxes": valid_box_count,
            "best_gt_index": best["gt_index"],
            "points_inside": best["points_inside"],
            "points_outside": best["points_outside"],
            "inside_ratio_percent": best["inside_ratio_percent"],
            "center_distance_m": best["center_distance_m"],
            "geometry_error_m": best["geometry_error_m"],
            "status": status,
        })

        print(
            f"  Boxes accepted: {valid_box_count}"
            f" | Best candidate GT: {best['gt_index']}"
            f" | Inside: {best['points_inside']}/{number_of_points}"
            f" ({best['inside_ratio_percent']:.2f}%)"
        )

    # ========================================================
    # SAVE CSV REPORTS
    # ========================================================

    summary_columns = [
        "frame_id", "car_cloud", "ground_truth_file",
        "input_points", "finite_points", "sparse_cloud",
        "valid_boxes", "best_gt_index", "points_inside",
        "points_outside", "inside_ratio_percent",
        "center_distance_m", "geometry_error_m", "status",
    ]

    candidate_columns = [
        "frame_id", "car_cloud", "gt_index", "finite_points",
        "points_inside", "points_outside", "inside_ratio_percent",
        "center_distance_m", "geometry_error_m",
    ]

    warning_columns = [
        "frame_id", "car_cloud", "warning", "details",
    ]

    pd.DataFrame(
        summary_rows, columns=summary_columns
    ).to_csv(SUMMARY_FILE, index=False)

    pd.DataFrame(
        candidate_rows, columns=candidate_columns
    ).to_csv(CANDIDATES_FILE, index=False)

    pd.DataFrame(
        warning_rows, columns=warning_columns
    ).to_csv(WARNINGS_FILE, index=False)

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 68)
    print("TASK 3 FINISHED")
    print("=" * 68)
    print(f"Car clouds evaluated: {len(summary_rows)}")
    print(f"Cloud-box candidate records: {len(candidate_rows)}")
    print(f"Warnings: {len(warning_rows)}")

    print(f"\nPer-car summary: {SUMMARY_FILE}")
    print(f"All candidates:  {CANDIDATES_FILE}")
    print(f"Warnings:        {WARNINGS_FILE}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)

        print("\nStatus counts:")
        print(summary["status"].value_counts().to_string())

        usable = summary[
            summary["status"].isin(["OK", "SPARSE_CLOUD"])
        ]

        if not usable.empty:
            print(
                "\nMean best-candidate inside ratio: "
                f"{usable['inside_ratio_percent'].mean():.2f}%"
            )
            print(
                "Clouds with at least one point inside a box: "
                f"{(usable['points_inside'] > 0).sum()}"
                f"/{len(usable)}"
            )

    print(
        "\nInterpretation: these are diagnostic point-in-box results. "
        "A best candidate is not automatically a confirmed object match."
    )


if __name__ == "__main__":
    main()
