
from pathlib import Path
import csv
import re

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# STAGE 3 - TASK 6: BATCH PROCESS SELECTED FRAMES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "Stage_1" / "Images"
LIDAR_DIR = (
    ROOT / "data_3d_raw"
    / "2013_05_28_drive_0000_sync"
    / "velodyne_points"
    / "data"
)
CALIB_DIR = ROOT / "calibration"

RESULTS_DIR = ROOT / "Stage_3" / "results"
SEG_DIR = RESULTS_DIR / "YOLOv8_Segmentation"
VALIDATION_DIR = RESULTS_DIR / "Mask_Validation"
ASSOCIATION_DIR = RESULTS_DIR / "LiDAR_Mask_Association"

MODEL_NAME = "yolov8x-seg.pt"
CAR_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.25

# None processes every image with a matching LiDAR scan.
# Example: FRAME_FILTER = ["0000000100"]
FRAME_FILTER = None

COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (128, 0, 255),
]

VALIDATION_FIELDS = [
    "frame",
    "car_id",
    "mask_file",
    "width",
    "height",
    "binary_values_only",
    "foreground_pixels",
]


def read_numeric_fields(path):
    fields = {}
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            nums = re.findall(pattern, value)

            if nums:
                fields[key.strip()] = np.array(
                    [float(v) for v in nums],
                    dtype=np.float64,
                )

    return fields


def load_transform(path):
    text = Path(path).read_text(encoding="utf-8")
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    values = np.array(
        [float(v) for v in re.findall(pattern, text)],
        dtype=np.float64,
    )

    if values.size != 12:
        raise ValueError(f"Expected 12 transform values in {path}")

    transform = np.eye(4)
    transform[:3, :] = values.reshape(3, 4)
    return transform


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_frame(model, image_path, lidar_path, R_rect, P_rect,
                  T_velo_to_cam):

    frame = image_path.stem
    image = cv2.imread(str(image_path))

    if image is None:
        return {"frame": frame, "status": "image_read_failed"}

    height, width = image.shape[:2]

    # ---------- Output directories ----------
    seg_frame_dir = SEG_DIR / frame
    masks_dir = seg_frame_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    assoc_frame_dir = ASSOCIATION_DIR / frame
    clouds_dir = assoc_frame_dir / "car_point_clouds"
    clouds_dir.mkdir(parents=True, exist_ok=True)

    # ---------- YOLOv8 segmentation ----------
    result = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    annotated = image.copy()
    detections = []
    masks = []

    if result.boxes is not None and result.masks is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        raw_masks = result.masks.data.cpu().numpy()

        car_id = 0

        for i, class_id in enumerate(classes):
            if class_id != CAR_CLASS_ID:
                continue

            car_id += 1
            x1, y1, x2, y2 = boxes[i].astype(int)

            mask = cv2.resize(
                raw_masks[i],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            binary = (mask > 0.5).astype(np.uint8) * 255

            mask_name = f"car_{car_id:02d}.png"
            cv2.imwrite(str(masks_dir / mask_name), binary)
            masks.append((car_id, mask_name, binary))

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            color = COLORS[(car_id - 1) % len(COLORS)]
            cv2.drawContours(annotated, contours, -1, color, 2)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                1,
            )
            cv2.putText(
                annotated,
                f"Car {car_id}: {confidences[i]:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

            detections.append({
                "frame": frame,
                "car_id": car_id,
                "class_id": int(class_id),
                "confidence": round(float(confidences[i]), 4),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "mask_pixels": int(np.count_nonzero(binary)),
                "mask_file": mask_name,
            })

    cv2.imwrite(
        str(seg_frame_dir / f"{frame}_yolo_cars.png"),
        annotated,
    )

    save_csv(
        seg_frame_dir / "detections.csv",
        detections,
        [
            "frame", "car_id", "class_id", "confidence",
            "x1", "y1", "x2", "y2", "mask_pixels", "mask_file",
        ],
    )

    # ---------- Validate and save masks ----------
    validation_rows = []

    for car_id, mask_name, mask in masks:
        unique_values = set(np.unique(mask).tolist())

        valid = (
            mask.shape == (height, width)
            and unique_values.issubset({0, 255})
            and np.count_nonzero(mask) > 0
        )

        validation_rows.append({
            "frame": frame,
            "car_id": car_id,
            "mask_file": mask_name,
            "width": width,
            "height": height,
            "binary_values_only": valid,
            "foreground_pixels": int(np.count_nonzero(mask)),
        })

    # One validation CSV per frame, including empty frames.
    save_csv(
        VALIDATION_DIR / f"{frame}_mask_validation.csv",
        validation_rows,
        VALIDATION_FIELDS,
    )

    # ---------- Load and project LiDAR ----------
    raw = np.fromfile(lidar_path, dtype=np.float32)

    if raw.size % 4 != 0:
        return {"frame": frame, "status": "invalid_lidar_file"}

    lidar_all = raw.reshape(-1, 4)
    original_indices = np.arange(len(lidar_all))

    finite = np.isfinite(lidar_all[:, :3]).all(axis=1)
    lidar = lidar_all[finite]
    original_indices = original_indices[finite]

    xyz = lidar[:, :3].astype(np.float64)
    xyz_h = np.column_stack((xyz, np.ones(len(xyz))))

    cam = (T_velo_to_cam @ xyz_h.T).T[:, :3]
    rect = (R_rect @ cam.T).T
    rect_h = np.column_stack((rect, np.ones(len(rect))))

    projected = (P_rect @ rect_h.T).T
    depth = projected[:, 2]

    in_front = depth > 1e-6
    pixels = np.full((len(xyz), 2), np.nan)

    pixels[in_front] = (
        projected[in_front, :2] / depth[in_front, None]
    )

    inside = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    positions = np.flatnonzero(inside)
    pixel_xy = np.rint(pixels[inside]).astype(int)

    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, width - 1)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, height - 1)

    # ---------- Associate LiDAR points with masks ----------
    overlay = image.copy()
    association_rows = []

    for car_id, mask_name, mask in masks:
        binary = mask > 0

        belongs = binary[pixel_xy[:, 1], pixel_xy[:, 0]]
        assigned_positions = positions[belongs]

        # Columns: x, y, z, reflectance, original LiDAR point index.
        cloud = np.column_stack((
            lidar[assigned_positions],
            original_indices[assigned_positions],
        ))

        cloud_path = clouds_dir / f"car_{car_id:02d}_points.npy"
        np.save(cloud_path, cloud.astype(np.float64))

        color = COLORS[(car_id - 1) % len(COLORS)]
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 1)

        association_rows.append({
            "frame": frame,
            "car_id": car_id,
            "mask_file": mask_name,
            "mask_pixels": int(np.count_nonzero(binary)),
            "associated_lidar_points": len(cloud),
            "cloud_file": cloud_path.name,
        })

    # Show a sample of projected LiDAR points.
    display_count = min(12000, len(positions))

    if display_count:
        chosen = np.linspace(
            0,
            len(positions) - 1,
            display_count,
            dtype=int,
        )

        for u, v in pixel_xy[chosen]:
            cv2.circle(
                overlay,
                (int(u), int(v)),
                1,
                (255, 255, 255),
                -1,
            )

    cv2.imwrite(
        str(assoc_frame_dir / f"{frame}_association_overlay.png"),
        overlay,
    )

    save_csv(
        assoc_frame_dir / "association_summary.csv",
        association_rows,
        [
            "frame", "car_id", "mask_file", "mask_pixels",
            "associated_lidar_points", "cloud_file",
        ],
    )

    return {
        "frame": frame,
        "status": "completed",
        "cars_detected": len(detections),
        "lidar_points_in_image": int(len(positions)),
        "associated_points": int(sum(
            row["associated_lidar_points"]
            for row in association_rows
        )),
    }


def main():
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ASSOCIATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load calibration.
    perspective = read_numeric_fields(CALIB_DIR / "perspective.txt")
    R_rect = perspective["R_rect_00"].reshape(3, 3)
    P_rect = perspective["P_rect_00"].reshape(3, 4)

    T_cam_to_velo = load_transform(
        CALIB_DIR / "calib_cam_to_velo.txt"
    )
    T_velo_to_cam = np.linalg.inv(T_cam_to_velo)

    # Find images.
    images = sorted(IMAGE_DIR.glob("*.png"))

    if FRAME_FILTER is not None:
        images = [p for p in images if p.stem in FRAME_FILTER]

    if not images:
        raise FileNotFoundError(
            f"No PNG images found in {IMAGE_DIR}"
        )

    model = YOLO(MODEL_NAME)
    batch_rows = []

    for image_path in images:
        frame = image_path.stem
        lidar_path = LIDAR_DIR / f"{frame}.bin"

        if not lidar_path.exists():
            print(f"Skipping {frame}: matching LiDAR scan not found")

            batch_rows.append({
                "frame": frame,
                "status": "missing_lidar",
                "cars_detected": 0,
                "lidar_points_in_image": 0,
                "associated_points": 0,
            })
            continue

        try:
            row = process_frame(
                model,
                image_path,
                lidar_path,
                R_rect,
                P_rect,
                T_velo_to_cam,
            )
            print(row)
            batch_rows.append(row)

        except Exception as error:
            print(f"Failed {frame}: {error}")

            batch_rows.append({
                "frame": frame,
                "status": f"error: {error}",
                "cars_detected": 0,
                "lidar_points_in_image": 0,
                "associated_points": 0,
            })

    # Save batch summary.
    save_csv(
        RESULTS_DIR / "batch_processing_summary.csv",
        batch_rows,
        [
            "frame", "status", "cars_detected",
            "lidar_points_in_image", "associated_points",
        ],
    )

    # Save one combined validation report for convenient review.
    all_validation_rows = []

    for validation_file in sorted(VALIDATION_DIR.glob("*_mask_validation.csv")):
        if validation_file.name == "all_frames_mask_validation.csv":
            continue

        with open(validation_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            all_validation_rows.extend(reader)

    save_csv(
        VALIDATION_DIR / "all_frames_mask_validation.csv",
        all_validation_rows,
        VALIDATION_FIELDS,
    )

    print("\nBatch processing finished.")
    print(f"Frames considered: {len(batch_rows)}")
    print(f"Summary: {RESULTS_DIR / 'batch_processing_summary.csv'}")
    print(f"Mask validation: {VALIDATION_DIR}")
    print(f"Mask records: {len(all_validation_rows)}")


if __name__ == "__main__":
    main()
