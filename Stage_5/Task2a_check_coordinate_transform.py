
from pathlib import Path

import numpy as np


# ============================================================
# STAGE 5 - TASK 2A
# LiDAR-to-Camera Coordinate Transform Verification
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = (
    ROOT / "calibration" / "calib_cam_to_velo.txt"
)

FRAME_ID = "0000000100"
CAR_ID = "car_01"

CLOUD_FILE = (
    ROOT
    / "Stage_3"
    / "results"
    / "LiDAR_Mask_Association"
    / FRAME_ID
    / "car_point_clouds"
    / f"{CAR_ID}_points.npy"
)


def load_transform(calibration_file):
    """Load the 3x4 camera-to-LiDAR calibration matrix."""
    values = np.loadtxt(calibration_file, dtype=np.float64)

    if values.size != 12:
        raise ValueError(
            f"Expected 12 calibration values, got {values.size}"
        )

    camera_to_lidar = np.eye(4, dtype=np.float64)
    camera_to_lidar[:3, :] = values.reshape(3, 4)

    return camera_to_lidar


def transform_points(points, transform):
    """Transform N x 3 points using a 4x4 homogeneous matrix."""
    points_h = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )

    transformed = (transform @ points_h.T).T

    return transformed[:, :3]


def main():
    print("=" * 65)
    print("STAGE 5 - TASK 2A")
    print("LiDAR-to-Camera Coordinate Transform Verification")
    print("=" * 65)

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    if not CLOUD_FILE.exists():
        raise FileNotFoundError(
            f"Point cloud file not found: {CLOUD_FILE}"
        )

    # 1. Load the camera-to-LiDAR calibration matrix.
    camera_to_lidar = load_transform(CALIBRATION_FILE)

    print("\n1. Camera-to-LiDAR transformation matrix:")
    print(camera_to_lidar)

    # 2. Calculate the inverse transformation.
    lidar_to_camera = np.linalg.inv(camera_to_lidar)

    print("\n2. LiDAR-to-camera transformation matrix:")
    print(lidar_to_camera)

    # 3. Verify that the matrices are inverses.
    identity_check = camera_to_lidar @ lidar_to_camera
    inverse_error = np.max(
        np.abs(identity_check - np.eye(4))
    )

    print("\n3. Inverse transformation check:")
    print(identity_check)
    print(f"Maximum identity error: {inverse_error:.10e}")

    if inverse_error > 1e-8:
        raise ValueError(
            "Transformation inverse check failed."
        )

    # 4. Load the saved Stage 3 car point cloud.
    cloud = np.load(CLOUD_FILE, allow_pickle=False)

    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(
            f"Unexpected point cloud shape: {cloud.shape}"
        )

    lidar_xyz = np.asarray(cloud[:, :3], dtype=np.float64)
    lidar_xyz = lidar_xyz[np.isfinite(lidar_xyz).all(axis=1)]

    if len(lidar_xyz) == 0:
        raise ValueError("No finite LiDAR points found.")

    # 5. Transform LiDAR points into camera coordinates.
    camera_xyz = transform_points(lidar_xyz, lidar_to_camera)

    # 6. Transform back to LiDAR coordinates.
    reconstructed_lidar_xyz = transform_points(
        camera_xyz, camera_to_lidar
    )

    round_trip_errors = np.linalg.norm(
        reconstructed_lidar_xyz - lidar_xyz, axis=1
    )

    print("\n4. Point-cloud transformation results:")
    print(f"Frame ID: {FRAME_ID}")
    print(f"Car ID: {CAR_ID}")
    print(f"Finite points: {len(lidar_xyz)}")

    print("\nLiDAR XYZ - first 5 points:")
    print(np.round(lidar_xyz[:5], 4))

    print("\nCamera XYZ - first 5 transformed points:")
    print(np.round(camera_xyz[:5], 4))

    print("\nCamera-coordinate range:")
    print("Minimum:", np.round(camera_xyz.min(axis=0), 4))
    print("Maximum:", np.round(camera_xyz.max(axis=0), 4))

    print("\n5. Round-trip verification:")
    print(
        "Maximum reconstruction error: "
        f"{round_trip_errors.max():.10e} m"
    )
    print(
        "Mean reconstruction error: "
        f"{round_trip_errors.mean():.10e} m"
    )

    print("\nTASK 2A COMPLETED")
    print(
        "The matrix inversion and round-trip transformation "
        "have been checked."
    )
    print(
        "Note: this verifies the mathematics of the transform, "
        "not the physical correctness of the calibration direction "
        "or the coordinate convention."
    )


if __name__ == "__main__":
    main()
