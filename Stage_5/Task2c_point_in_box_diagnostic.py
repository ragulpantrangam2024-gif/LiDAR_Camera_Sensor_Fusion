
from pathlib import Path
import json
import numpy as np

# ============================================================
# STAGE 5 - TASK 2: POINT-IN-BOX DIAGNOSTIC
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CALIB_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"
BOX_DIR = ROOT / "bboxes_3D_cam0"
ASSOCIATION_DIR = (
    ROOT / "Stage_3" / "results" / "LiDAR_Mask_Association"
)

FRAME_ID = "0000000100"
CAR_ID = "car_01"
TOLERANCE = 1e-3

CAR_FILE = (
    ASSOCIATION_DIR / FRAME_ID /
    "car_point_clouds" / f"{CAR_ID}_points.npy"
)
BOX_FILE = BOX_DIR / "BBoxes_100.json"


def load_transform(path):
    """Load the 3x4 camera-to-LiDAR calibration matrix."""
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values, got {values.size}"
        )

    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def get_box_geometry(corners):
    """
    Use the observed corner connectivity:
    corner 0 -> corners 1, 2, and 5.

    Verify that the other corners agree with the inferred box.
    """
    corners = np.asarray(corners, dtype=float)

    if corners.shape != (8, 3):
        raise ValueError(f"Expected (8, 3) corners, got {corners.shape}")

    if not np.isfinite(corners).all():
        raise ValueError("Corners contain non-finite values")

    origin = corners[0]

    edge1 = corners[1] - origin
    edge2 = corners[2] - origin
    edge3 = corners[5] - origin

    basis = np.column_stack([edge1, edge2, edge3])

    if abs(np.linalg.det(basis)) < 1e-8:
        raise ValueError("Degenerate box geometry")

    # Expected corner order under the observed connectivity.
    expected = np.array([
        origin,
        origin + edge1,
        origin + edge2,
        origin + edge1 + edge2,
        origin + edge1 + edge3,
        origin + edge3,
        origin + edge2 + edge3,
        origin + edge1 + edge2 + edge3,
    ])

    # Match every expected vertex to a unique supplied corner.
    remaining = list(corners)

    for point in expected:
        distances = [
            np.linalg.norm(point - candidate)
            for candidate in remaining
        ]
        nearest = int(np.argmin(distances))

        if distances[nearest] > TOLERANCE:
            raise ValueError(
                "Corners do not match the observed connectivity"
            )

        remaining.pop(nearest)

    return origin, basis


def points_inside_box(points, corners):
    """Return a Boolean mask identifying points inside the box."""
    origin, basis = get_box_geometry(corners)

    # Convert camera-frame points into box-local coordinates.
    local = np.linalg.solve(basis, (points - origin).T).T

    return np.all(
        (local >= -1e-6) & (local <= 1.0 + 1e-6),
        axis=1
    )


def main():
    print("=" * 65)
    print("STAGE 5 - TASK 2: POINT-IN-BOX DIAGNOSTIC")
    print("=" * 65)

    for path in (CALIB_FILE, CAR_FILE, BOX_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Invert camera-to-LiDAR to obtain LiDAR-to-camera.
    T_cam_to_velo = load_transform(CALIB_FILE)
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    cloud = np.load(CAR_FILE)

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"Unexpected cloud shape: {cloud.shape}")

    xyz_velo = cloud[:, :3]
    xyz_velo = xyz_velo[np.isfinite(xyz_velo).all(axis=1)]

    homogeneous = np.column_stack([
        xyz_velo,
        np.ones(len(xyz_velo))
    ])
    xyz_cam = (T_velo_to_cam @ homogeneous.T).T[:, :3]

    with BOX_FILE.open("r", encoding="utf-8") as f:
        boxes = json.load(f)

    print(f"\nFrame: {FRAME_ID}")
    print(f"Car cloud: {CAR_FILE.name}")
    print(f"Finite LiDAR points: {len(xyz_cam)}")
    print(f"Ground-truth boxes: {len(boxes)}")
    print("Transformed XYZ minimum:", np.round(xyz_cam.min(axis=0), 3))
    print("Transformed XYZ maximum:", np.round(xyz_cam.max(axis=0), 3))

    results = []
    skipped = 0

    for box in boxes:
        index = box.get("index", "unknown")

        try:
            corners = np.asarray(box["corners_cam0"], dtype=float)
            inside = points_inside_box(xyz_cam, corners)

            count_inside = int(inside.sum())
            center = corners.mean(axis=0)
            distance = float(
                np.linalg.norm(np.median(xyz_cam, axis=0) - center)
            )

            results.append({
                "index": index,
                "inside": count_inside,
                "outside": len(xyz_cam) - count_inside,
                "distance": distance,
                "center": center,
            })

        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            skipped += 1
            print(f"Skipped box {index}: {exc}")

    results.sort(key=lambda item: (-item["inside"], item["distance"]))

    print("\n" + "=" * 65)
    print("TOP 10 GROUND-TRUTH BOXES")
    print("=" * 65)

    for result in results[:10]:
        print(
            f"GT index={result['index']}, "
            f"inside={result['inside']}, "
            f"outside={result['outside']}, "
            f"center_distance={result['distance']:.2f} m, "
            f"center={np.round(result['center'], 2)}"
        )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Boxes evaluated: {len(results)}")
    print(f"Boxes skipped: {skipped}")

    if results:
        print(
            "Maximum points inside a box:",
            max(item["inside"] for item in results)
        )

    print(
        "\nDiagnostic only: a high point count does not prove "
        "that the ground-truth box matches this detected car."
    )


if __name__ == "__main__":
    main()
