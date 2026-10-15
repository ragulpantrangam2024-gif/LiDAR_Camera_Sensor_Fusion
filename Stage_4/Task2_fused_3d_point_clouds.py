
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STAGE 4 - TASK 2: FUSED 3D POINT CLOUDS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLOUD_DIR = (
    ROOT / "Stage_3" / "results"
    / "LiDAR_Mask_Association"
)

RESULTS_DIR = ROOT / "Stage_4" / "results"
OUTPUT_DIR = RESULTS_DIR / "Fused_3D_Point_Clouds"

FRAME_FILTER = None  # Example: ["0000000100"]

COLORS = [
    "red", "limegreen", "blue", "gold",
    "magenta", "cyan", "darkorange", "purple",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(
        p for p in CLOUD_DIR.iterdir() if p.is_dir()
    )

    if FRAME_FILTER is not None:
        frame_dirs = [
            p for p in frame_dirs if p.name in FRAME_FILTER
        ]

    summary_rows = []

    for frame_dir in frame_dirs:
        frame = frame_dir.name
        point_cloud_dir = frame_dir / "car_point_clouds"

        if not point_cloud_dir.exists():
            print(f"Skipping {frame}: no point-cloud directory")
            continue

        cloud_paths = sorted(
            point_cloud_dir.glob("car_*_points.npy")
        )

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        total_points = 0
        clouds_plotted = 0

        for i, cloud_path in enumerate(cloud_paths):
            cloud = np.load(cloud_path)

            if cloud.ndim != 2 or cloud.shape[1] < 3:
                print(f"Skipping invalid cloud: {cloud_path.name}")
                continue

            points = cloud[:, :3]
            if len(points) == 0:
                continue

            # Use a consistent color for each car within the frame.
            color = COLORS[i % len(COLORS)]

            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=2,
                c=color,
                label=f"Car {i + 1} ({len(points)} pts)",
            )

            total_points += len(points)
            clouds_plotted += 1

        ax.set_title(f"Frame {frame}: Fused 3D LiDAR Point Clouds")
        ax.set_xlabel("LiDAR X (m)")
        ax.set_ylabel("LiDAR Y (m)")
        ax.set_zlabel("LiDAR Z (m)")

        if clouds_plotted:
            ax.legend(loc="upper right", fontsize=8)

        output_path = OUTPUT_DIR / f"{frame}_fused_3d.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        summary_rows.append({
            "frame": frame,
            "car_clouds_plotted": clouds_plotted,
            "total_point_assignments": total_points,
            "output_image": output_path.name,
        })

        print({
            "frame": frame,
            "car_clouds_plotted": clouds_plotted,
            "total_point_assignments": total_points,
            "output": str(output_path),
        })

    summary_path = OUTPUT_DIR / "fused_3d_summary.csv"

    with open(
        summary_path, "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "frame",
            "car_clouds_plotted",
            "total_point_assignments",
            "output_image",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nStage 4 Task 2 completed.")
    print(f"Frames processed: {len(summary_rows)}")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
