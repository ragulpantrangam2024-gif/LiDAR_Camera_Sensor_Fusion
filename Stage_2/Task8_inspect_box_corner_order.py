
from pathlib import Path
import json
import numpy as np

# ============================================================
# STAGE 2 - TASK 8: INSPECT 3D BOX CORNER ORDER
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "bboxes_3D_cam0" / "BBoxes_100.json"

# Inspect foreground objects and one distant object for comparison
OBJECT_IDS = [131, 145, 210, 215, 216, 218]


def load_objects(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("objects", "annotations", "boxes", "bboxes"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unexpected JSON structure. Check Task 6's JSON loader."
    )


def main():
    print("=" * 70)
    print("STAGE 2 - TASK 8: INSPECT 3D BOX CORNER ORDER")
    print("=" * 70)

    objects = load_objects(JSON_PATH)
    object_map = {obj["index"]: obj for obj in objects}

    # Candidate connectivity based on the documented corner convention
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 5), (1, 4), (2, 7), (3, 6),
    ]

    for object_id in OBJECT_IDS:
        obj = object_map.get(object_id)

        if obj is None:
            print(f"\nObject {object_id}: not found")
            continue

        corners = np.asarray(
            obj["corners_cam0"], dtype=np.float64
        )

        if corners.shape != (8, 3):
            print(f"\nObject {object_id}: invalid shape {corners.shape}")
            continue

        print(f"\n{'-' * 70}")
        print(f"OBJECT ID: {object_id}")
        print("Corner coordinates in cam0:")
        print("Corner      X          Y          Z")

        for i, (x, y, z) in enumerate(corners):
            print(f"{i:>6}  {x:10.3f} {y:10.3f} {z:10.3f}")

        print("\nCandidate edge lengths:")
        lengths = []

        for a, b in edges:
            length = float(np.linalg.norm(corners[a] - corners[b]))
            lengths.append((a, b, length))
            print(f"{a}-{b}: {length:.3f} m")

        print("\nCoordinate ranges:")
        for axis, name in enumerate(("X", "Y", "Z")):
            print(
                f"{name}: {corners[:, axis].min():.3f} "
                f"to {corners[:, axis].max():.3f} m"
            )

        # Opposite faces should generally have corresponding edge
        # lengths. Print lengths only; do not assume the ordering
        # is correct based on length alone.
        print("\nFace-edge length comparison:")
        for i in range(4):
            a, b, length = lengths[i]
            opposite_length = lengths[i + 4][2]
            print(
                f"Edges {a}-{b} and "
                f"{edges[i + 4][0]}-{edges[i + 4][1]}: "
                f"{length:.3f} m vs {opposite_length:.3f} m"
            )

    print("\n" + "=" * 70)
    print("Inspection complete. No annotation data was modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
