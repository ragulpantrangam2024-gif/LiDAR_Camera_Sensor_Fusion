
from pathlib import Path
import re
import json

import numpy as np
import open3d as o3d


# ============================================================
# STAGE 4 - TASK 4
# Interactive multi-frame LiDAR point-cloud viewer
# Keys: N = next frame, P = previous frame, Q = quit
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLOUD_ROOT = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)
GT_ROOT = ROOT / "bboxes_3D_cam0"
CALIB_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"

FRAME_IDS = [
    "0000000100", "0000000250", "0000000360",
    "0000000570", "0000000644", "0000000729",
    "0000000947", "0000001134", "0000001280",
    "0000001338", "0000001461", "0000001597",
    "0000001791", "0000001866", "0000002033",
    "0000002098", "0000002449", "0000002717",
    "0000002903", "0000002939",
]

COLORS = [
    [1.0, 0.1, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.3, 1.0],
    [1.0, 0.8, 0.0],
    [1.0, 0.1, 0.9],
    [0.0, 0.9, 0.9],
    [1.0, 0.5, 0.0],
    [0.6, 0.3, 1.0],
]

FRAME_INDEX = 0
VISUALIZER = None
CURRENT_GEOMETRIES = []
FRAME_INFO = {}


def load_transform(path):
    text = Path(path).read_text(encoding="utf-8")
    values = re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    values = np.asarray([float(v) for v in values])

    if values.size != 12:
        raise ValueError(
            f"Expected 12 transform values in {path}"
        )

    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def find_corner_arrays(value, results):
    """Find nested arrays containing exactly eight XYZ corners."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_lower = key.lower()

            if isinstance(child, (list, tuple)):
                try:
                    arr = np.asarray(child, dtype=float)
                    if arr.shape == (8, 3):
                        results.append((key, arr))
                        continue
                except (ValueError, TypeError):
                    pass

            find_corner_arrays(child, results)

    elif isinstance(value, list):
        try:
            arr = np.asarray(value, dtype=float)
            if arr.shape == (8, 3):
                results.append(("corners", arr))
                return
        except (ValueError, TypeError):
            pass

        for child in value:
            find_corner_arrays(child, results)


def load_ground_truth_boxes(frame):
    """Load GT cuboids and transform camera-frame corners to LiDAR."""
    candidates = [
        GT_ROOT / f"BBoxes_{int(frame)}.json",
        GT_ROOT / f"{frame}.json",
        GT_ROOT / f"bboxes_{frame}.json",
    ]

    json_path = next((p for p in candidates if p.exists()), None)

    if json_path is None:
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"Could not read GT file: {json_path}")
        return []

    found = []
    find_corner_arrays(data, found)

    # Deduplicate identical corner arrays.
    unique_boxes = []
    seen = set()

    for _, corners in found:
        signature = tuple(np.round(corners.flatten(), 5))
        if signature not in seen:
            seen.add(signature)
            unique_boxes.append(corners)

    T_cam_to_velo = load_transform(CALIB_FILE)
    box_lines = []

    # Corner ordering from the project report.
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    for corners_cam in unique_boxes:
        corners_h = np.column_stack(
            (corners_cam, np.ones(8))
        )
        corners_lidar = (
            T_cam_to_velo @ corners_h.T
        ).T[:, :3]

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(
            corners_lidar
        )
        lines.lines = o3d.utility.Vector2iVector(edges)
        lines.paint_uniform_color([1.0, 1.0, 1.0])
        box_lines.append(lines)

    return box_lines


def build_frame_geometries(frame):
    geometries = []
    cloud_dir = CLOUD_ROOT / frame / "car_point_clouds"

    cloud_count = 0
    point_count = 0

    if cloud_dir.exists():
        cloud_paths = sorted(
            cloud_dir.glob("car_*_points.npy")
        )

        for i, path in enumerate(cloud_paths):
            data = np.load(path)

            if data.ndim != 2 or data.shape[1] < 3:
                continue

            xyz = data[:, :3]
            xyz = xyz[np.isfinite(xyz).all(axis=1)]

            if len(xyz) == 0:
                continue

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(xyz)
            cloud.paint_uniform_color(COLORS[i % len(COLORS)])

            geometries.append(cloud)
            cloud_count += 1
            point_count += len(xyz)

    boxes = load_ground_truth_boxes(frame)
    geometries.extend(boxes)

    # Coordinate frame makes orientation easier to understand.
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0,
        origin=[0.0, 0.0, 0.0],
    )
    geometries.append(axis)

    FRAME_INFO[frame] = {
        "car_clouds": cloud_count,
        "points": point_count,
        "gt_boxes": len(boxes),
    }

    return geometries


def show_frame(index, reset_camera=False):
    global CURRENT_GEOMETRIES, FRAME_INDEX

    FRAME_INDEX = index % len(FRAME_IDS)
    frame = FRAME_IDS[FRAME_INDEX]

    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.remove_geometry(
            geometry, reset_bounding_box=False
        )

    CURRENT_GEOMETRIES = build_frame_geometries(frame)

    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.add_geometry(
            geometry, reset_bounding_box=False
        )

    if reset_camera:
        VISUALIZER.reset_view_point(True)

    VISUALIZER.update_renderer()

    info = FRAME_INFO[frame]
    print(
        f"\nFrame {FRAME_INDEX + 1}/{len(FRAME_IDS)}: {frame}\n"
        f"Car clouds: {info['car_clouds']}\n"
        f"Point assignments: {info['points']}\n"
        f"GT boxes loaded: {info['gt_boxes']}"
    )


def next_frame(_vis):
    show_frame(FRAME_INDEX + 1, reset_camera=True)
    return False


def previous_frame(_vis):
    show_frame(FRAME_INDEX - 1, reset_camera=True)
    return False


def main():
    global VISUALIZER, CURRENT_GEOMETRIES

    if not CLOUD_ROOT.exists():
        raise FileNotFoundError(
            f"Point-cloud root not found: {CLOUD_ROOT}"
        )

    VISUALIZER = o3d.visualization.VisualizerWithKeyCallback()
    VISUALIZER.create_window(
        window_name="KITTI-360 Interactive LiDAR Fusion",
        width=1400,
        height=900,
    )

    CURRENT_GEOMETRIES = build_frame_geometries(
        FRAME_IDS[FRAME_INDEX]
    )

    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.add_geometry(geometry)

    VISUALIZER.register_key_callback(ord("N"), next_frame)
    VISUALIZER.register_key_callback(ord("P"), previous_frame)

    frame = FRAME_IDS[FRAME_INDEX]
    info = FRAME_INFO[frame]

    print("Interactive KITTI-360 viewer")
    print("N: next frame | P: previous frame | Q: quit")
    print("Mouse: rotate, zoom and pan")
    print(f"Frames available: {len(FRAME_IDS)}")
    print(f"Current frame: {frame}")
    print(f"GT boxes loaded: {info['gt_boxes']}")

    VISUALIZER.run()
    VISUALIZER.destroy_window()


if __name__ == "__main__":
    main()
