
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
    parameters = {}

    with file_path.open("r", encoding="utf-8-sig") as file:
        content = file.read().strip()

    if not content:
        raise ValueError(f"Calibration file is empty: {file_path}")

    # Format 1: plain numeric values, such as calib_cam_to_velo.txt
    if ":" not in content:
        values = np.fromstring(content.replace("\n", " "), sep=" ")

        if values.size == 12:
            parameters["transform_matrix"] = values.reshape(3, 4)
        else:
            raise ValueError(
                f"Expected 12 values for a 3x4 transformation matrix, "
                f"but found {values.size} in {file_path.name}"
            )

        return parameters

    # Format 2: named parameters, such as perspective.txt
    for line in content.splitlines():
        line = line.strip()

        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)

        try:
            values = np.array(
                [float(item) for item in value.split()],
                dtype=np.float64
            )
        except ValueError:
            continue

        if values.size:
            parameters[key.strip()] = values

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
