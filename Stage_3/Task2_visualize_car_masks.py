
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# STAGE 3 - TASK 2: VISUALIZE ALL CAR MASKS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FRAME = "0000000100"

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
MASKS_DIR = (
    ROOT / "Stage_3" / "results" / FRAME / "masks"
)

OUTPUT_DIR = ROOT / "Stage_3" / "results" / FRAME
OUTPUT_PATH = OUTPUT_DIR / f"{FRAME}_colored_masks.png"

ALPHA = 0.45


def main():
    print("=" * 60)
    print("STAGE 3 - TASK 2: VISUALIZE CAR MASKS")
    print("=" * 60)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    mask_paths = sorted(MASKS_DIR.glob("car_*.png"))

    if not mask_paths:
        raise FileNotFoundError(
            f"No individual masks found in: {MASKS_DIR}"
        )

    height, width = image.shape[:2]
    overlay = image.copy()

    # Distinct colors in OpenCV BGR format
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (128, 0, 255),
        (255, 128, 0),
        (128, 255, 0),
    ]

    print(f"Frame: {FRAME}")
    print(f"Image size: {width} x {height}")
    print(f"Masks found: {len(mask_paths)}")

    # Blend each mask with the original image.
    for i, mask_path in enumerate(mask_paths):
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            print(f"Warning: could not read {mask_path.name}")
            continue

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = mask > 0
        color = colors[i % len(colors)]

        # Color only the pixels belonging to this mask.
        overlay[binary] = (
            ALPHA * np.array(color)
            + (1 - ALPHA) * overlay[binary]
        ).astype(np.uint8)

        # Draw the mask boundary.
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            1,
            cv2.LINE_AA,
        )

        # Label each mask near its centroid.
        moments = cv2.moments(binary.astype(np.uint8))

        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            cv2.putText(
                overlay,
                f"Car {i + 1}",
                (cx, max(cy - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        print(
            f"{mask_path.name}: "
            f"{int(np.count_nonzero(binary))} mask pixels"
        )

    if not cv2.imwrite(str(OUTPUT_PATH), overlay):
        raise RuntimeError(
            f"Could not save output: {OUTPUT_PATH}"
        )

    print("-" * 60)
    print(f"Saved visualization: {OUTPUT_PATH}")
    print("Task 2 completed.")


if __name__ == "__main__":
    main()
