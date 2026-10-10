
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STAGE 3 - TASK 5: VISUALIZE INDIVIDUAL CAR POINT CLOUDS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FRAME = "0000000100"

CLOUDS_DIR = (
    ROOT / "Stage_3" / "results" / FRAME / "lidar_car_clouds"
)
OUTPUT_DIR = ROOT / "Stage_3" / "results" / FRAME
OUTPUT_PATH = OUTPUT_DIR / f"{FRAME}_car_point_clouds_3d.png"

def main():
    print("=" * 65)
    print("STAGE 3 - TASK 5: 3D CAR POINT CLOUD VISUALIZATION")
    print("=" * 65)

    cloud_paths = sorted(CLOUDS_DIR.glob("car_*_points.npy"))

    if not cloud_paths:
        raise FileNotFoundError(
            f"No point-cloud files found in {CLOUDS_DIR}"
        )

    number_of_clouds = len(cloud_paths)

    # Two columns; add rows as needed.
    columns = 2
    rows = (number_of_clouds + columns - 1) // columns

    fig = plt.figure(figsize=(14, 5 * rows))
    summary = []

    for i, cloud_path in enumerate(cloud_paths):
        cloud = np.load(cloud_path)

        if cloud.ndim != 2 or cloud.shape[1] < 4:
            print(f"Skipping invalid cloud: {cloud_path.name}")
            continue

        xyz = cloud[:, :3]
        reflectance = cloud[:, 3]

        ax = fig.add_subplot(rows, columns, i + 1, projection="3d")

        if len(xyz) > 0:
            scatter = ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=reflectance,
                cmap="viridis",
                s=3,
            )

            fig.colorbar(
                scatter,
                ax=ax,
                shrink=0.65,
                label="Reflectance",
            )

            ax.set_xlim(
                np.percentile(xyz[:, 0], [1, 99])
            )
            ax.set_ylim(
                np.percentile(xyz[:, 1], [1, 99])
            )
            ax.set_zlim(
                np.percentile(xyz[:, 2], [1, 99])
            )

        ax.set_title(
            f"Car {i + 1} — {len(xyz)} points"
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        summary.append({
            "file": cloud_path.name,
            "points": len(xyz),
        })

        print(f"{cloud_path.name}: {len(xyz)} points")

    fig.suptitle(
        f"LiDAR Point Clouds Associated with Car Masks — {FRAME}",
        fontsize=14,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("-" * 65)
    print(f"Clouds visualized: {len(summary)}")
    print(f"Saved figure: {OUTPUT_PATH}")
    print("Task 5 completed.")


if __name__ == "__main__":
    main()
