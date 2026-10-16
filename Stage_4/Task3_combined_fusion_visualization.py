
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STAGE 4 - TASK 3: COMBINED FUSION VISUALIZATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "Stage_1" / "Images"
SEG_DIR = ROOT / "Stage_3" / "results" / "YOLOv8_Segmentation"
CLOUD_DIR = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)
PROJECTION_DIR = (
    ROOT / "Stage_4" / "results"
    / "Instance_Colored_Projection"
)
POINT_CLOUD_3D_DIR = (
    ROOT / "Stage_4" / "results"
    / "Fused_3D_Point_Clouds"
)

OUTPUT_DIR = (
    ROOT / "Stage_4" / "results"
    / "Combined_Fusion_Visualization"
)

FRAME_FILTER = None  # Example: ["0000000100"]

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 165, 255), (128, 0, 255),
]


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if FRAME_FILTER is not None:
        image_paths = [
            p for p in image_paths if p.stem in FRAME_FILTER
        ]

    summary_rows = []

    for image_path in image_paths:
        frame = image_path.stem

        try:
            original = read_image(image_path)
            height, width = original.shape[:2]

            # ---------- Segmentation visualization ----------
            segmentation = original.copy()
            masks_dir = SEG_DIR / frame / "masks"
            mask_count = 0

            if masks_dir.exists():
                for i, mask_path in enumerate(
                    sorted(masks_dir.glob("car_*.png"))
                ):
                    mask = cv2.imread(
                        str(mask_path),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if mask is None:
                        continue

                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    binary = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        binary,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    color = COLORS[i % len(COLORS)]
                    cv2.drawContours(
                        segmentation, contours, -1, color, 2
                    )
                    mask_count += 1

            # ---------- LiDAR fusion overlay ----------
            overlay_path = (
                PROJECTION_DIR
                / f"{frame}_instance_colored_lidar.png"
            )

            if overlay_path.exists():
                fusion = read_image(overlay_path)
            else:
                fusion = original.copy()

            # ---------- Corresponding 3D point-cloud plot ----------
            plot_path = (
                POINT_CLOUD_3D_DIR / f"{frame}_fused_3d.png"
            )

            # Compose a four-panel comparison.
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            panels = [
                (axes[0, 0], original, "Original Camera Image"),
                (axes[0, 1], segmentation, "Car Instance Masks"),
                (axes[1, 0], fusion, "Instance-Colored LiDAR Overlay"),
            ]

            for ax, image, title in panels:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax.set_title(title)
                ax.axis("off")

            axes[1, 1].axis("off")

            if plot_path.exists():
                # Display the saved 3D plot as a panel.
                plot_image = plt.imread(plot_path)
                axes[1, 1].imshow(plot_image)
                axes[1, 1].set_title("Fused 3D LiDAR Point Clouds")
                axes[1, 1].axis("off")
            else:
                axes[1, 1].text(
                    0.5, 0.5,
                    "3D point-cloud plot unavailable",
                    ha="center", va="center",
                )

            fig.suptitle(
                f"Camera-LiDAR Fusion: Frame {frame}",
                fontsize=16,
            )
            fig.tight_layout()

            output_path = (
                OUTPUT_DIR / f"{frame}_combined_fusion.png"
            )
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Count existing per-car cloud files.
            cloud_dir = CLOUD_DIR / frame / "car_point_clouds"
            cloud_count = (
                len(list(cloud_dir.glob("car_*_points.npy")))
                if cloud_dir.exists() else 0
            )

            total_points = 0
            if cloud_dir.exists():
                for cloud_path in cloud_dir.glob("car_*_points.npy"):
                    cloud = np.load(cloud_path, mmap_mode="r")
                    if cloud.ndim == 2:
                        total_points += len(cloud)

            summary_rows.append({
                "frame": frame,
                "masks_visualized": mask_count,
                "car_cloud_files": cloud_count,
                "associated_point_assignments": total_points,
                "fusion_overlay_available": overlay_path.exists(),
                "3d_plot_available": plot_path.exists(),
                "output_file": output_path.name,
                "status": "completed",
            })

            print({
                "frame": frame,
                "masks_visualized": mask_count,
                "car_cloud_files": cloud_count,
                "associated_point_assignments": total_points,
                "output": str(output_path),
            })

        except Exception as error:
            print(f"Failed {frame}: {error}")
            summary_rows.append({
                "frame": frame,
                "masks_visualized": 0,
                "car_cloud_files": 0,
                "associated_point_assignments": 0,
                "fusion_overlay_available": False,
                "3d_plot_available": False,
                "output_file": "",
                "status": f"error: {error}",
            })

    summary_path = OUTPUT_DIR / "combined_fusion_summary.csv"

    fields = [
        "frame",
        "masks_visualized",
        "car_cloud_files",
        "associated_point_assignments",
        "fusion_overlay_available",
        "3d_plot_available",
        "output_file",
        "status",
    ]

    with open(
        summary_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nStage 4 Task 3 completed.")
    print(f"Frames considered: {len(summary_rows)}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
