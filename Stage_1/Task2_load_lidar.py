
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

LIDAR_PATH = (
    PROJECT_DIR
    / "data_3d_raw"
    / "2013_05_28_drive_0000_sync"
    / "velodyne_points"
    / "data"
    / "0000000100.bin"
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check the input file
if not LIDAR_PATH.exists():
    raise FileNotFoundError(f"LiDAR file not found: {LIDAR_PATH}")

# Each point contains x, y, z and reflectance (4 float32 values)
raw_data = np.fromfile(LIDAR_PATH, dtype=np.float32)

if raw_data.size == 0:
    raise ValueError("The LiDAR file is empty.")

if raw_data.size % 4 != 0:
    raise ValueError(
        f"Unexpected number of values: {raw_data.size}. "
        "Expected four values per point."
    )

points = raw_data.reshape(-1, 4)

xyz = points[:, :3]
reflectance = points[:, 3]

# Keep only points with finite coordinates
valid = np.isfinite(xyz).all(axis=1)
xyz = xyz[valid]
reflectance = reflectance[valid]

print("LiDAR point cloud loaded successfully!")
print(f"Filename: {LIDAR_PATH.name}")
print(f"Number of valid points: {len(xyz)}")
print(f"XYZ array shape: {xyz.shape}")
print(f"Reflectance array shape: {reflectance.shape}")
print(f"X range: {xyz[:, 0].min():.2f} to {xyz[:, 0].max():.2f} m")
print(f"Y range: {xyz[:, 1].min():.2f} to {xyz[:, 1].max():.2f} m")
print(f"Z range: {xyz[:, 2].min():.2f} to {xyz[:, 2].max():.2f} m")

# Save the XYZ point coordinates for inspection
output_path = RESULTS_DIR / "0000000100_xyz.npy"
np.save(output_path, xyz)
print(f"Saved XYZ points to: {output_path}")

# Top-down view: X forward, Y lateral
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    xyz[:, 0],
    xyz[:, 1],
    c=reflectance,
    s=0.2,
    cmap="viridis"
)
plt.colorbar(scatter, label="Reflectance")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("KITTI-360 LiDAR Point Cloud - Frame 0000000100")
plt.axis("equal")
plt.tight_layout()

plot_path = RESULTS_DIR / "0000000100_lidar_topdown.png"
plt.savefig(plot_path, dpi=200)
plt.show()

print(f"Saved visualization to: {plot_path}")
