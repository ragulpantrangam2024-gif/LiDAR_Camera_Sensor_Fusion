Stage 1 — Dataset and Data Loading

Task 1: Load Camera Image

Objective: Load and inspect a KITTI-360 camera image using Python and OpenCV.

Implementation:

Loaded frame 0000000100.png using OpenCV.

Checked image dimensions, channels and data type.

Converted BGR to RGB for visualization with Matplotlib.

Saved the loaded image in the results/ directory.

Script: Task1_load_image.py

Output: results/0000000100_loaded.png

Libraries: OpenCV, Matplotlib, pathlib.

Status: Implemented; verify by running the script.

### Task 4 — Load Ground-Truth 3D Bounding Boxes

**Objective:** Load and validate the KITTI-360 ground-truth 3D bounding-box annotations.

**Implementation:**
- Loaded JSON annotations from `bboxes_3D_cam0`.
- Extracted each object's index and eight `corners_cam0` coordinates.
- Validated the shape and numerical values of each 3D box.
- Calculated coordinate minima and maxima for inspection.
- Exported validated annotations to JSON and a summary to CSV.

**Outputs:**
- `results/0000000100_ground_truth_validated.json`
- `results/0000000100_ground_truth_summary.csv`

**Outcome:** Ground-truth annotations are prepared for later LiDAR point-in-box testing and 3D detection evaluation.