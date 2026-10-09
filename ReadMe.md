# KITTI-360 dataset sample

This is sample of [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) dataset. It contains the 20 frames of Cam 1 and 2, as well as the point clouds of the Velodyne LiDAR.

For how the data is structured, please consult the official [documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php)

You can also take a look at the [GitHub-Repo](https://github.com/autonomousvision/kitti360Scripts) to find some helper functions (like projecting the point cloud to the image space and how to load data) already implemented.

The folder bboxes_3D_cam0 was created by us. It contains a JSON-File for every frame. Each object has an index, which is a unique ID for a car, as well as 8 corners for a 3D bounding box. The bounding box is given in the coordinate space of cam0.

The corner points are ordered like this:

     Top view:
       4 -------- 5
      /          /
     7 -------- 6

     Bottom view:
       0 -------- 1
      /          /
     3 -------- 2

Task 2: Load LiDAR Point Cloud

Objective: Load a KITTI-360 Velodyne binary point cloud and inspect its 3D coordinates.

Implementation:

Read the .bin file using NumPy.

Reshaped the data into four values per point: X, Y, Z and reflectance.

Filtered non-finite XYZ coordinates.

Printed point-cloud statistics and coordinate ranges.

Saved the XYZ coordinates and a top-down visualization.

Script: Task2_load_lidar.py

Outputs:

results/0000000100_xyz.npy

results/0000000100_lidar_topdown.png

Libraries: NumPy, Matplotlib, pathlib.

Status: Implemented; verify by running the script.

## Task 3: Load Calibration Parameters

**Objective:** Load and inspect KITTI-360 calibration files required for camera–LiDAR sensor fusion.

**Implementation:**
- Read the camera-to-LiDAR calibration file.
- Read perspective projection and rectification parameters.
- Inspect camera-to-pose calibration entries.
- Parse colon-separated numeric parameters.
- Display parameter names, value counts and common matrix dimensions.

**Script:** `Task3_load_calibration.py`

**Input files:**
- `calibration/calib_cam_to_velo.txt`
- `calibration/perspective.txt`
- `calibration/calib_cam_to_pose.txt`

**Libraries:** NumPy, pathlib.

**Status:** Implemented; verify against the actual calibration files.