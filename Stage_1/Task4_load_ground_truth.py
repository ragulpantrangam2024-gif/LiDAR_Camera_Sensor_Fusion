
import json
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
GT_DIR = PROJECT_DIR / "bboxes_3D_cam0"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FRAME_ID = "0000000100"


def inspect_structure(data, indent=0):
    """Print the structure of a JSON annotation."""
    prefix = " " * indent

    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} keys")

        for key, value in list(data.items())[:10]:
            print(f"{prefix}Key: {key}")

            if isinstance(value, (dict, list)):
                inspect_structure(value, indent + 2)
            else:
                print(f"{prefix}  Value: {value}")

    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items")

        if data:
            print(f"{prefix}Inspecting first item:")
            inspect_structure(data[0], indent + 2)

    else:
        print(f"{prefix}Value: {data}")


def find_corner_arrays(data, path="root"):
    """Find arrays that appear to contain eight 3D corners."""
    if isinstance(data, dict):
        for key, value in data.items():
            yield from find_corner_arrays(value, f"{path}.{key}")

    elif isinstance(data, list):
        # Candidate: eight corners, each containing three coordinates
        if (
            len(data) == 8
            and all(
                isinstance(point, (list, tuple))
                and len(point) == 3
                and all(isinstance(v, (int, float)) for v in point)
                for point in data
            )
        ):
            yield path, data
        else:
            for index, value in enumerate(data):
                yield from find_corner_arrays(value, f"{path}[{index}]")


def main():
    if not GT_DIR.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {GT_DIR}"
        )

    json_files = sorted(GT_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON annotation files found in {GT_DIR}"
        )

    # Prefer an annotation matching the selected image frame.
    matching_files = [
        path for path in json_files
        if path.stem == FRAME_ID
        or path.stem.endswith(FRAME_ID)
    ]

    json_path = matching_files[0] if matching_files else json_files[0]

    print("=" * 60)
    print("STAGE 1 — TASK 4: LOAD GROUND-TRUTH 3D BOXES")
    print("=" * 60)
    print(f"Annotation file: {json_path.name}")
    print(f"Total JSON files available: {len(json_files)}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("\n--- JSON structure ---")
    inspect_structure(data)

    candidates = list(find_corner_arrays(data))

    print("\n--- Candidate 3D corner arrays ---")
    print(f"Arrays containing eight 3D points: {len(candidates)}")

    for index, (path, corners) in enumerate(candidates[:5], start=1):
        print(f"\nCandidate {index}")
        print(f"Location: {path}")
        for corner_index, point in enumerate(corners, start=1):
            print(f"Corner {corner_index}: {point}")

    # Save the original parsed annotation for reproducibility.
    output_path = RESULTS_DIR / f"{json_path.stem}_ground_truth.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"\nParsed annotation saved to: {output_path}")
    print("\nTask 4 inspection completed.")


if __name__ == "__main__":
    main()
