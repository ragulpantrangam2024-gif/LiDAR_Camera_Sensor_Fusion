
from pathlib import Path
import csv
import re

import numpy as np


# ============================================================
# STAGE 4 - TASK 5
# Fusion Statistics and Export
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE3_RESULTS = ROOT / "Stage_3" / "results"
ASSOCIATION_ROOT = (
    STAGE3_RESULTS / "LiDAR_Mask_Association"
)

# Output directory
OUTPUT_DIR = (
    ROOT / "Stage_4" / "results" / "Fusion_Statistics"
)

# Initial reporting threshold; adjust if required.
SPARSE_POINT_THRESHOLD = 20

FRAME_IDS = [
    "0000000100",
    "0000000250",
    "0000000360",
    "0000000570",
    "0000000644",
    "0000000729",
    "0000000947",
    "0000001134",
    "0000001280",
    "0000001338",
    "0000001461",
    "0000001597",
    "0000001791",
    "0000001866",
    "0000002033",
    "0000002098",
    "0000002449",
    "0000002717",
    "0000002903",
    "0000002939",
]


def find_cloud_directory(frame):
    """Find the per-car point-cloud folder for a frame."""
    candidates = [
        ASSOCIATION_ROOT / frame / "car_point_clouds",
        ASSOCIATION_ROOT / frame / "lidar_car_clouds",
        STAGE3_RESULTS / frame / "car_point_clouds",
        STAGE3_RESULTS / frame / "lidar_car_clouds",
    ]

    for directory in candidates:
        if directory.exists():
            return directory

    # Support minor differences in folder layout.
    for root in [
        ASSOCIATION_ROOT / frame,
        STAGE3_RESULTS / frame,
    ]:
        if root.exists():
            matches = list(root.rglob("car_*_points.npy"))
            if matches:
                return matches[0].parent

    return None


def load_car_clouds(frame):
    """Return per-car point counts and validation statuses."""
    directory = find_cloud_directory(frame)

    if directory is None:
        return [], "missing_cloud_files"

    rows = []
    files = sorted(directory.glob("car_*_points.npy"))

    for path in files:
        match = re.search(r"car_(\d+)_points\.npy", path.name)
        car_id = match.group(1) if match else path.stem

        try:
            data = np.load(path, mmap_mode="r")

            if data.ndim != 2 or data.shape[1] < 3:
                rows.append({
                    "frame_id": frame,
                    "car_id": car_id,
                    "point_count": 0,
                    "cloud_status": "invalid_array",
                    "file": str(path.relative_to(ROOT)),
                })
                continue

            xyz = np.asarray(data[:, :3])
            valid = np.isfinite(xyz).all(axis=1)
            point_count = int(valid.sum())

            if point_count == 0:
                status = "empty"
            elif point_count < SPARSE_POINT_THRESHOLD:
                status = "sparse"
            else:
                status = "sufficient_points"

            rows.append({
                "frame_id": frame,
                "car_id": car_id,
                "point_count": point_count,
                "cloud_status": status,
                "file": str(path.relative_to(ROOT)),
            })

        except (OSError, ValueError) as error:
            rows.append({
                "frame_id": frame,
                "car_id": car_id,
                "point_count": 0,
                "cloud_status": f"read_error: {error}",
                "file": str(path.relative_to(ROOT)),
            })

    return rows, "available"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    per_car_rows = []
    exception_rows = []

    for frame in FRAME_IDS:
        car_rows, folder_status = load_car_clouds(frame)
        per_car_rows.extend(car_rows)

        total_points = sum(
            row["point_count"] for row in car_rows
        )
        car_count = len(car_rows)

        empty_count = sum(
            row["cloud_status"] == "empty"
            for row in car_rows
        )
        sparse_count = sum(
            row["cloud_status"] == "sparse"
            for row in car_rows
        )

        if folder_status != "available":
            frame_status = folder_status
        elif car_count == 0:
            frame_status = "no_car_clouds"
            exception_rows.append({
                "frame_id": frame,
                "car_id": "",
                "point_count": 0,
                "cloud_status": "no_car_clouds",
                "file": "",
            })
        else:
            frame_status = "available"

        per_frame_rows.append({
            "frame_id": frame,
            "car_clouds_found": car_count,
            "total_point_assignments": total_points,
            "empty_clouds": empty_count,
            "sparse_clouds": sparse_count,
            "sparse_threshold": SPARSE_POINT_THRESHOLD,
            "frame_status": frame_status,
        })

        for row in car_rows:
            if row["cloud_status"] in (
                "empty", "sparse", "invalid_array"
            ) or row["cloud_status"].startswith("read_error"):
                exception_rows.append(row)

        print({
            "frame": frame,
            "car_clouds": car_count,
            "point_assignments": total_points,
            "empty_clouds": empty_count,
            "sparse_clouds": sparse_count,
            "status": frame_status,
        })

    write_csv(
        OUTPUT_DIR / "per_frame_summary.csv",
        per_frame_rows,
        [
            "frame_id",
            "car_clouds_found",
            "total_point_assignments",
            "empty_clouds",
            "sparse_clouds",
            "sparse_threshold",
            "frame_status",
        ],
    )

    write_csv(
        OUTPUT_DIR / "per_car_summary.csv",
        per_car_rows,
        [
            "frame_id",
            "car_id",
            "point_count",
            "cloud_status",
            "file",
        ],
    )

    write_csv(
        OUTPUT_DIR / "empty_and_sparse_clouds.csv",
        exception_rows,
        [
            "frame_id",
            "car_id",
            "point_count",
            "cloud_status",
            "file",
        ],
    )

    print("\nStage 4 Task 5 completed.")
    print(f"Frames analyzed: {len(per_frame_rows)}")
    print(f"Car-cloud records: {len(per_car_rows)}")
    print(f"Empty/sparse/problem records: {len(exception_rows)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
