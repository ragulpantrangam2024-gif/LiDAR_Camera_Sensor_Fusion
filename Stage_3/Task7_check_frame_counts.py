
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "Stage_1" / "Images"
LIDAR_DIR = (
    ROOT / "data_3d_raw"
    / "2013_05_28_drive_0000_sync"
    / "velodyne_points"
    / "data"
)

# Collect available image and LiDAR frame IDs
images = {p.stem for p in IMAGE_DIR.glob("*") if p.suffix.lower() == ".png"}
lidar = {p.stem for p in LIDAR_DIR.glob("*.bin")}

matching = images & lidar
images_without_lidar = images - lidar
lidar_without_images = lidar - images

print(f"Images found: {len(images)}")
print(f"LiDAR files found: {len(lidar)}")
print(f"Frames with both image and LiDAR: {len(matching)}")

print("\nImage frame IDs:")
for frame in sorted(images):
    print(frame)

print("\nImages without matching LiDAR:")
for frame in sorted(images_without_lidar):
    print(frame)

print("\nLiDAR files without matching images:")
for frame in sorted(lidar_without_images):
    print(frame)
