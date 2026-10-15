LiDAR–Camera Sensor Fusion

KITTI-360 Dataset Sample

This project uses a sample of the KITTI-360 dataset. It contains 20 selected camera frames from Cam 1 and Cam 2, along with the corresponding Velodyne LiDAR point clouds.

For details about the dataset structure, refer to the official KITTI-360 documentation.

You can also explore the KITTI-360 Scripts repository for helper functions related to data loading and projecting LiDAR point clouds into image space.

3D Bounding Boxes

The bboxes_3D_cam0/ directory was created for this project. It contains one JSON annotation file per frame. Each object includes a unique index identifying a car and eight corner points defining its 3D bounding box in the Cam 0 coordinate system.

The corner points are ordered as follows:

Top view

4 -------- 5
/          /
7 -------- 6

Bottom view

0 -------- 1
/          /
3 -------- 2

Input Calibration Files

calibration/calib_cam_to_velo.txt

calibration/perspective.txt

calibration/calib_cam_to_pose.txt

Libraries

NumPy

pathlib

Status

Implementation completed. Calibration and projection results should be verified against the supplied calibration files.