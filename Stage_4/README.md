Stage 4 — LiDAR–Camera Sensor Fusion

This stage combines camera-based car detections, LiDAR point clouds, and 3D ground-truth bounding boxes to visualize and analyze sensor-fusion results across selected KITTI-360 frames.

Tasks

Task 1 — Load Fusion Data: Load the required frame data, LiDAR point clouds, and available annotations.

Task 2 — 3D Point Cloud Visualization: Visualize LiDAR points in 3D to understand the scene structure.

Task 3 — Ground-Truth Box Visualization: Display 3D ground-truth bounding boxes together with the LiDAR point cloud.

Task 4 — Interactive 3D Point Cloud Viewer: Navigate through frames and inspect LiDAR points and ground-truth boxes interactively.

Task 4B — Interactive Bird’s-Eye View: View the full scene from above, including raw LiDAR points, associated car point clouds, and ground-truth boxes where available.

Running Task 4B

From the project root, run:

python Stage_4/Task4B_birds_eye_viewer.py

Use the interactive viewer to navigate through frames:

N — Next frame

P — Previous frame

S — Save the current screenshot

Left mouse drag — Rotate

Mouse wheel — Zoom

Right mouse drag — Pan

Q — Quit

Output

Results are saved under Stage_4/results/, including bird’s-eye-view screenshots.

Note: Automatically saved views provide a consistent overview, but some frames may require manual rotation, zooming, and panning for better interpretation. Use the interactive viewer to inspect these frames and save a clearer screenshot with S.

Some frames may contain few or no associated car LiDAR points, or may have incomplete ground-truth annotations. These cases should be interpreted accordingly.