from pathlib import Path
import re
import json
import time

from PIL import Image
import numpy as np
import open3d as o3d


# ============================================================
# STAGE 4 - TASK 4B
# Full-scene interactive bird's-eye LiDAR fusion viewer
#
# Background: light gray full LiDAR scan
# Car points: dark, saturated instance colors
# GT boxes: red wireframes
#
# Controls:
#   N = next frame
#   P = previous frame
#   S = save current view
#   Q = quit
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

LIDAR_DIR = (
    ROOT / "data_3d_raw" / "2013_05_28_drive_0000_sync"
    / "velodyne_points" / "data"
)
CLOUD_ROOT = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)
GT_ROOT = ROOT / "bboxes_3D_cam0"
CALIB_FILE = ROOT / "calibration" / "calib_cam_to_velo.txt"
OUTPUT_DIR = ROOT / "Stage_4" / "results" / "Birds_Eye_View"

# The 20 frame IDs listed for this project/report.
EXPECTED_FRAMES = [
    "0000000100", "0000000250", "0000000360", "0000000570",
    "0000000644", "0000000729", "0000000947", "0000001134",
    "0000001280", "0000001338", "0000001461", "0000001597",
    "0000001791", "0000001866", "0000002033", "0000002098",
    "0000002449", "0000002717", "0000002903", "0000002939",
]

# Light gray background scan; dark saturated car points.
BACKGROUND_COLOR = [0.78, 0.78, 0.78]
CAR_COLORS = [
    [0.90, 0.00, 0.00],  # red
    [0.00, 0.55, 0.00],  # green
    [0.00, 0.10, 0.90],  # blue
    [0.75, 0.48, 0.00],  # ochre
    [0.75, 0.00, 0.65],  # magenta
    [0.00, 0.55, 0.65],  # teal
    [0.85, 0.25, 0.00],  # orange
    [0.35, 0.00, 0.75],  # purple
]
GT_BOX_COLOR = [1.0, 0.0, 0.0]

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
POINT_SIZE = 3.0
MAX_BACKGROUND_POINTS = 150000

FRAME_IDS = []
FRAME_INDEX = 0
VISUALIZER = None
CURRENT_GEOMETRIES = []
FRAME_INFO = {}


def load_transform(path):
    """Read a 3x4 transform from a KITTI-style calibration text file."""
    text = Path(path).read_text(encoding="utf-8")
    values = re.findall(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text
    )
    values = np.asarray([float(v) for v in values], dtype=float)
    if values.size != 12:
        raise ValueError(
            f"Expected 12 numeric transform values in {path}; "
            f"found {values.size}."
        )
    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def load_car_clouds(frame):
    """Load per-car point arrays and preserve their original LiDAR indices."""
    results_root = ROOT / "Stage_3" / "results"
    candidate_roots = [
        CLOUD_ROOT / frame,
        results_root / frame,
        results_root / "LiDAR_Mask_Association" / frame,
    ]

    paths = []
    seen = set()
    for root in candidate_roots:
        if root.exists():
            for path in root.rglob("car_*_points.npy"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    paths.append(path)
                    seen.add(resolved)
    paths.sort(key=lambda p: p.name)

    if not paths:
        return [], 0, set(), f"No car_*_points.npy files found for {frame}."

    geometries = []
    total_points = 0
    original_indices = set()
    for index, path in enumerate(paths):
        try:
            data = np.load(path)
        except (OSError, ValueError) as exc:
            print(f"Warning: cannot read {path}: {exc}")
            continue

        if data.ndim != 2 or data.shape[1] < 3:
            print(f"Warning: unexpected array shape in {path}: {data.shape}")
            continue

        xyz = np.asarray(data[:, :3], dtype=np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if len(xyz) == 0:
            continue

        # Stage 3 Task 4 stores columns [x, y, z, reflectance, original_index].
        # Collect those source indices so the same points can be removed from
        # the gray background and rendered only in their instance color.
        if data.shape[1] >= 5:
            idx_values = np.asarray(data[:, 4])[finite]
            idx_values = idx_values[np.isfinite(idx_values)]
            original_indices.update(idx_values.astype(np.int64).tolist())

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.paint_uniform_color(CAR_COLORS[index % len(CAR_COLORS)])
        geometries.append(cloud)
        total_points += len(xyz)

    error = None if geometries else f"No valid car points loaded for {frame}."
    return geometries, total_points, original_indices, error


def load_background_cloud(frame, excluded_indices=None):
    """Load the full scan, excluding points drawn separately in car colors."""
    path = LIDAR_DIR / f"{frame}.bin"
    if not path.exists():
        return None, 0, f"Missing LiDAR file: {path.name}"

    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        return None, 0, f"Invalid four-float LiDAR format: {path.name}"

    scan = raw.reshape(-1, 4)
    xyz_all = scan[:, :3].astype(np.float64)
    raw_count = int(np.isfinite(xyz_all).all(axis=1).sum())

    valid = np.isfinite(xyz_all).all(axis=1)
    # Crucial visibility fix: car points are also present in the raw scan.
    # If they are drawn twice at identical coordinates, gray points can hide
    # the colored points due to depth buffering. Remove them from background.
    if excluded_indices:
        idx = np.fromiter(excluded_indices, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(scan))]
        valid[idx] = False

    xyz = xyz_all[valid]
    distance = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(distance > 1.0) & (distance < 100.0)]

    if len(xyz) > MAX_BACKGROUND_POINTS:
        ids = np.linspace(0, len(xyz) - 1, MAX_BACKGROUND_POINTS, dtype=int)
        xyz = xyz[ids]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.paint_uniform_color(BACKGROUND_COLOR)
    return cloud, raw_count, None


def find_gt_file(frame):
    candidates = [
        GT_ROOT / f"BBoxes_{int(frame)}.json",
        GT_ROOT / f"BBoxes_{frame}.json",
        GT_ROOT / f"{frame}.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_ground_truth_boxes(frame):
    """
    Read the eight camera-coordinate corners and transform them into
    LiDAR coordinates using calib_cam_to_velo.txt.

    Accepts common corner field names to handle annotation-file variants.
    """
    path = find_gt_file(frame)
    if path is None:
        return [], f"Missing GT JSON for frame {frame}"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        transform = load_transform(CALIB_FILE)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], f"Cannot load GT/calibration for {frame}: {exc}"

    if isinstance(payload, dict):
        objects = payload.get("objects")
        if objects is None:
            objects = list(payload.values())
    elif isinstance(payload, list):
        objects = payload
    else:
        return [], f"Unexpected GT JSON structure: {path.name}"

    # Common field variants; use the first one present in each object.
    corner_fields = (
        "corners_cam0", "corners", "corners_3d",
        "bbox_3d", "corners_camera",
    )
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        corners = None
        for field in corner_fields:
            if field in obj:
                corners = obj[field]
                break
        if corners is None:
            continue

        try:
            corners = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError):
            continue

        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            continue

        homogeneous = np.column_stack((corners, np.ones(8)))
        lidar_corners = (transform @ homogeneous.T).T[:, :3]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(lidar_corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(GT_BOX_COLOR)
        boxes.append(line_set)

    if not boxes:
        return [], (
            f"No compatible 8x3 corner arrays found in {path.name}; "
            "check the JSON corner field name."
        )
    return boxes, None


def build_frame_geometries(frame):
    geometries = []

    # Load colored subsets first, so their source indices can be excluded from
    # the gray full-scene cloud. This prevents gray points occluding them.
    car_geometries, associated_count, associated_indices, car_error = load_car_clouds(frame)
    background, raw_count, background_error = load_background_cloud(
        frame, excluded_indices=associated_indices
    )

    if background is not None:
        geometries.append(background)
    geometries.extend(car_geometries)

    gt_geometries, gt_error = load_ground_truth_boxes(frame)
    geometries.extend(gt_geometries)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0.0, 0.0, 0.0]
    )
    geometries.append(axes)

    FRAME_INFO[frame] = {
        "raw_lidar_points": raw_count,
        "background_points": len(background.points) if background is not None else 0,
        "car_clouds": len(car_geometries),
        "associated_points": associated_count,
        "excluded_background_points": len(associated_indices),
        "gt_boxes": len(gt_geometries),
        "warnings": [
            msg for msg in (background_error, car_error, gt_error) if msg
        ],
    }
    return geometries


def set_birds_eye_camera():
    """
    Fit the current frame to the viewport with a consistent top-down view.
    The view is reset after the frame's geometries are added, so the camera
    does not inherit a zoom/pan/rotation from the previous frame.
    """
    view = VISUALIZER.get_view_control()
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_lookat([0.0, 0.0, 0.0])
    view.set_zoom(0.75)


def save_current_screenshot():
    """
    Capture the currently rendered Open3D window after allowing the renderer
    to finish updating. Save the rendered framebuffer, not a separately
    reconstructed plot, so the saved image matches the interactive view.
    """
    if VISUALIZER is None or not FRAME_IDS:
        return

    frame = FRAME_IDS[FRAME_INDEX]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{frame}_birds_eye_view.png"

    # Give Open3D a few event/render cycles. Capturing immediately after
    # changing frames can save a partially updated scene on some Windows setups.
    for _ in range(3):
        VISUALIZER.poll_events()
        VISUALIZER.update_renderer()
        time.sleep(0.08)

    try:
        framebuffer = np.asarray(
            VISUALIZER.capture_screen_float_buffer(do_render=True),
            dtype=np.float32
        )
        if framebuffer.ndim != 3 or framebuffer.shape[2] < 3:
            raise RuntimeError(
                f"Unexpected framebuffer shape: {framebuffer.shape}"
            )

        rgb = np.clip(framebuffer[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)
        print(
            f"Screenshot saved: {path} "
            f"(frame {frame}, size {rgb.shape[1]}x{rgb.shape[0]})"
        )
    except Exception as exc:
        print(f"Screenshot capture failed for {frame}: {exc}")


def show_frame(index):
    global FRAME_INDEX, CURRENT_GEOMETRIES
    FRAME_INDEX = index % len(FRAME_IDS)
    frame = FRAME_IDS[FRAME_INDEX]

    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.remove_geometry(geometry, reset_bounding_box=False)

    CURRENT_GEOMETRIES = build_frame_geometries(frame)
    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.add_geometry(geometry, reset_bounding_box=False)

    VISUALIZER.reset_view_point(True)
    set_birds_eye_camera()
    VISUALIZER.poll_events()
    VISUALIZER.update_renderer()

    info = FRAME_INFO[frame]
    print(f"\nFrame {FRAME_INDEX + 1}/{len(FRAME_IDS)}: {frame}")
    print(f"  Raw LiDAR points: {info['raw_lidar_points']}")
    print(f"  Gray background points: {info['background_points']}")
    print(f"  Car clouds: {info['car_clouds']}")
    print(f"  Colored car points: {info['associated_points']}")
    print(f"  Removed from gray background: {info['excluded_background_points']}")
    print(f"  GT boxes: {info['gt_boxes']}")
    for warning in info["warnings"]:
        print(f"  WARNING: {warning}")

    # Save the current rendered frame after geometry and camera updates.
    # This ensures all 20 frame screenshots are generated as the user navigates.
    save_current_screenshot()


def next_frame(_visualizer):
    show_frame(FRAME_INDEX + 1)
    return False


def previous_frame(_visualizer):
    show_frame(FRAME_INDEX - 1)
    return False


def save_frame_callback(_visualizer):
    save_current_screenshot()
    return False


def main():
    global VISUALIZER, CURRENT_GEOMETRIES, FRAME_IDS

    if not LIDAR_DIR.exists():
        raise FileNotFoundError(f"LiDAR directory not found:\n{LIDAR_DIR}")
    if not CALIB_FILE.exists():
        raise FileNotFoundError(f"Calibration file not found:\n{CALIB_FILE}")

    # Check all expected frames, rather than silently assuming every input exists.
    FRAME_IDS = EXPECTED_FRAMES.copy()
    missing_lidar = [f for f in FRAME_IDS if not (LIDAR_DIR / f"{f}.bin").exists()]
    results_root = ROOT / "Stage_3" / "results"
    missing_clouds = []
    for frame in FRAME_IDS:
        roots = [CLOUD_ROOT / frame, results_root / frame]
        has_clouds = any(
            root.exists() and any(root.rglob("car_*_points.npy"))
            for root in roots
        )
        if not has_clouds:
            missing_clouds.append(frame)

    print(f"Expected frames: {len(FRAME_IDS)}")
    print(f"LiDAR scans missing: {len(missing_lidar)}")
    if missing_lidar:
        print("Missing LiDAR frame IDs:", ", ".join(missing_lidar))
    print(f"Associated-cloud folders missing: {len(missing_clouds)}")
    if missing_clouds:
        print("Missing cloud frame IDs:", ", ".join(missing_clouds))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZER = o3d.visualization.VisualizerWithKeyCallback()
    VISUALIZER.create_window(
        window_name="KITTI-360 | Full Scene Bird's-Eye LiDAR Fusion",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )

    render = VISUALIZER.get_render_option()
    render.background_color = np.array([1.0, 1.0, 1.0])
    render.point_size = POINT_SIZE

    CURRENT_GEOMETRIES = build_frame_geometries(FRAME_IDS[FRAME_INDEX])
    for geometry in CURRENT_GEOMETRIES:
        VISUALIZER.add_geometry(geometry)

    VISUALIZER.reset_view_point(True)
    set_birds_eye_camera()
    VISUALIZER.poll_events()
    VISUALIZER.update_renderer()

    VISUALIZER.register_key_callback(ord("N"), next_frame)
    VISUALIZER.register_key_callback(ord("P"), previous_frame)
    VISUALIZER.register_key_callback(ord("S"), save_frame_callback)

    print("\nControls: N = next | P = previous | S = save | Q = quit")
    print("Mouse: left-drag rotate, wheel zoom, right-drag pan")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Manual navigation is enabled. Only the current frame is shown.")
    print("Press N/P to navigate one frame at a time; each displayed frame is saved after rendering. Press S to save again.")
    show_frame(0)
    VISUALIZER.run()
    VISUALIZER.destroy_window()

    # Verify output count after the viewer closes.
    saved = sorted(OUTPUT_DIR.glob("*_birds_eye_view.png"))
    saved_ids = {p.name.replace("_birds_eye_view.png", "") for p in saved}
    missing_outputs = [f for f in FRAME_IDS if f not in saved_ids]

    print("\nViewer closed.")
    print(f"Expected frames: {len(FRAME_IDS)}")
    print(f"Screenshots saved: {len(saved_ids)}")
    if missing_outputs:
        print("Frames without saved screenshots:", ", ".join(missing_outputs))
        print("Navigate to each missing frame; the viewer will save it after rendering.")
    else:
        print("All 20 frame screenshots are present.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
