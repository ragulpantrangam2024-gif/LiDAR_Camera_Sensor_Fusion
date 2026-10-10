
from pathlib import Path
import csv

import cv2
import numpy as np

# ============================================================
# STAGE 3 - TASK 3: VALIDATE CAR MASKS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FRAME = "0000000100"

IMAGE_PATH = ROOT / "Stage_1" / "Images" / f"{FRAME}.png"
MASKS_DIR = ROOT / "Stage_3" / "results" / FRAME / "masks"
RESULTS_DIR = ROOT / "Stage_3" / "results" / FRAME

SUMMARY_PATH = RESULTS_DIR / f"{FRAME}_mask_validation.csv"

def main():
    print("=" * 60)
    print("STAGE 3 - TASK 3: VALIDATE CAR MASKS")
    print("=" * 60)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    height, width = image.shape[:2]
    mask_paths = sorted(MASKS_DIR.glob("car_*.png"))

    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {MASKS_DIR}")

    rows = []
    all_valid = True

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"{mask_path.name}: FAILED to load")
            all_valid = False
            continue

        dimensions_ok = mask.shape == (height, width)
        unique_values = np.unique(mask)
        binary_ok = set(unique_values.tolist()).issubset({0, 255})
        foreground_pixels = int(np.count_nonzero(mask))

        valid = dimensions_ok and binary_ok and foreground_pixels > 0
        all_valid = all_valid and valid

        rows.append({
            "mask_file": mask_path.name,
            "width": mask.shape[1],
            "height": mask.shape[0],
            "dimensions_match": dimensions_ok,
            "binary_values_only": binary_ok,
            "foreground_pixels": foreground_pixels,
            "valid": valid,
        })

        print(
            f"{mask_path.name}: "
            f"dimensions={mask.shape[1]}x{mask.shape[0]}, "
            f"binary={binary_ok}, "
            f"foreground={foreground_pixels}, "
            f"valid={valid}"
        )

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mask_file", "width", "height",
                "dimensions_match", "binary_values_only",
                "foreground_pixels", "valid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 60)
    print(f"Image dimensions: {width} x {height}")
    print(f"Masks checked: {len(rows)}")
    print(f"All masks valid: {all_valid}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print("Task 3 completed.")


if __name__ == "__main__":
    main()
