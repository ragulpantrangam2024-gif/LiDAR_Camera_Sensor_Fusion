
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOX_FILE = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"

with BOX_FILE.open("r", encoding="utf-8") as f:
    boxes = json.load(f)

# Inspect the box that is closest to the transformed car cloud.
TARGET_INDEX = 218

box = next(
    (item for item in boxes if item.get("index") == TARGET_INDEX),
    None
)

if box is None:
    raise ValueError(f"Ground-truth box {TARGET_INDEX} not found")

corners = np.asarray(box["corners_cam0"], dtype=float)

print("=" * 65)
print("GROUND-TRUTH CORNER GEOMETRY")
print("=" * 65)
print("Frame: 0000000100")
print("GT box index:", TARGET_INDEX)
print("\nAll eight corners:")

for i, corner in enumerate(corners):
    print(f"Corner {i}: {np.round(corner, 6)}")

print("\nPairwise distances from each corner:")
distances = np.linalg.norm(
    corners[:, None, :] - corners[None, :, :],
    axis=2
)

for i in range(8):
    nearest = np.argsort(distances[i])
    nearest = [j for j in nearest if j != i][:3]
    print(
        f"Corner {i}: "
        + ", ".join(
            f"corner {j} = {distances[i, j]:.4f} m"
            for j in nearest
        )
    )

print("\nEdge-length candidates from corner 0:")
for i in range(1, 8):
    print(f"0 -> {i}: {distances[0, i]:.4f} m")

print("\nInspection complete.")
