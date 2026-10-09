
from pathlib import Path

import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CALIB_DIR = PROJECT_DIR / "calibration"

FILES_TO_LOAD = [
    "calib_cam_to_velo.txt",
    "perspective.txt",
    "calib_cam_to_pose.txt",
]


def load_calibration_file(file_path):
    """Read calibration entries and parse colon-separated numeric values."""
    parameters = {}

    with file_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or ":" not in line:
                continue

            key, value = line.split(":", 1)
            values = value.split()

            try:
                numbers = np.array(
                    [float(item) for item in values],
                    dtype=np.float64,
                )
            except ValueError:
                # Ignore non-numeric entries in this basic loader.
                continue

            if numbers.size:
                parameters[key.strip()] = numbers

    return parameters


def describe_parameter(key, values):
    """Display values and recognize common matrix sizes."""
    count = values.size

    if count == 9:
        shape = "(3, 3)"
    elif count == 12:
        shape = "(3, 4)"
    elif count == 16:
        shape = "(4, 4)"
    elif count == 3:
        shape = "(3,) — vector"
    else:
        shape = f"({count},) — numeric values"

    print(f"\nParameter: {key}")
    print(f"Number of values: {count}")
    print(f"Interpreted shape: {shape}")

    if count == 9:
        print(values.reshape(3, 3))
    elif count == 12:
        print(values.reshape(3, 4))
    elif count == 16:
        print(values.reshape(4, 4))
    else:
        print(values)


def main():
    print("STAGE 1 — TASK 3: LOAD CALIBRATION PARAMETERS")

    for filename in FILES_TO_LOAD:
        file_path = CALIB_DIR / filename

        print("\n" + "=" * 60)
        print(f"File: {filename}")
        print("=" * 60)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        parameters = load_calibration_file(file_path)

        if not parameters:
            print(
                "No colon-separated numeric parameters found. "
                "Inspect the file format before parsing it."
            )
            continue

        print(f"Parameters found: {len(parameters)}")

        for key, values in parameters.items():
            describe_parameter(key, values)


if __name__ == "__main__":
    main()
