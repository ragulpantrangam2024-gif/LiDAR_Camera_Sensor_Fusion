
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# STAGE 3 - TASK 1: YOLOv8 CAR SEGMENTATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FRAME = "0000000100"
IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
MODEL_NAME = "yolov8x-seg.pt"
CAR_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.25

RESULTS_DIR = ROOT / "Stage_3" / "results" / FRAME
MASKS_DIR = RESULTS_DIR / "masks"

MASKS_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_PATH = RESULTS_DIR / f"{FRAME}_yolo_cars.png"
CSV_PATH = RESULTS_DIR / f"{FRAME}_detections.csv"


def main():
    print("=" * 60)
    print("STAGE 3 - TASK 1: YOLOv8 CAR SEGMENTATION")
    print("=" * 60)

    # Load image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    print(f"Frame: {FRAME}")
    print(f"Image size: {width} x {height}")
    print(f"Loading model: {MODEL_NAME}")

    # Load pretrained segmentation model
    model = YOLO(MODEL_NAME)

    # Run segmentation
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]
    annotated = image.copy()
    detections = []

    if result.boxes is not None and result.masks is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        car_number = 0

        for i, class_id in enumerate(classes):
            # Keep only cars
            if class_id != CAR_CLASS_ID:
                continue

            car_number += 1
            confidence = float(confidences[i])

            x1, y1, x2, y2 = boxes[i].astype(int)

            # Resize mask to original image dimensions
            mask = cv2.resize(
                masks[i],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            # Save individual car mask
            mask_path = MASKS_DIR / f"car_{car_number:02d}.png"

            if not cv2.imwrite(str(mask_path), binary_mask):
                raise RuntimeError(
                    f"Could not save mask: {mask_path}"
                )

            # Draw mask boundary and detection box
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                annotated,
                contours,
                -1,
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            label = f"Car {car_number}: {confidence:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            detections.append({
                "car_id": car_number,
                "class_id": int(class_id),
                "confidence": round(confidence, 4),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "mask_pixels": int(np.count_nonzero(binary_mask)),
                "mask_file": mask_path.name,
            })

    # Save annotated image
    if not cv2.imwrite(str(ANNOTATED_PATH), annotated):
        raise RuntimeError(
            f"Could not save annotated image: {ANNOTATED_PATH}"
        )

    # Save detection summary
    fieldnames = [
        "car_id", "class_id", "confidence",
        "x1", "y1", "x2", "y2",
        "mask_pixels", "mask_file",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)

    print("\nResults")
    print("-" * 60)
    print(f"Cars detected: {len(detections)}")
    print(f"Annotated image: {ANNOTATED_PATH}")
    print(f"Individual masks: {MASKS_DIR}")
    print(f"Detection summary: {CSV_PATH}")
    print("\nTask 1 completed.")


if __name__ == "__main__":
    main()
