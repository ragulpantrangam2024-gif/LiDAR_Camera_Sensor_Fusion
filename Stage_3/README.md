Stage 3: YOLOv8 Segmentation and LiDAR Association

Overview

This stage uses YOLOv8 instance segmentation to detect cars in camera images and associates projected LiDAR points with each car's 2D segmentation mask. The objective is to generate individual car masks and corresponding LiDAR point clouds for subsequent camera–LiDAR fusion.

Tasks Completed

Task 1: YOLOv8 Car Segmentation

Applied YOLOv8 instance segmentation to detect cars in camera images.

Extracted individual masks and bounding boxes for detected cars.

Saved annotated images, binary masks, and detection details.

Task 2: Mask Visualization

Visualized individual car masks using different colors.

Inspected mask boundaries and overlapping regions.

Task 3: Mask Validation

Checked mask dimensions and binary pixel values.

Counted foreground pixels for each mask.

Saved per-frame validation CSV files and a combined validation report.

Task 4: LiDAR–Mask Association

Projected LiDAR points into the camera image using calibration parameters.

Selected LiDAR points whose projected pixels fall inside each car mask.

Saved individual point clouds and association summaries.

Task 5: 3D Point-Cloud Visualization

Visualized the LiDAR point clouds associated with individual car masks.

Inspected the distribution of points associated with detected cars.

Task 6: Batch Processing

Processed all 20 selected camera frames with matching LiDAR scans.

Automated segmentation, mask validation, and LiDAR-mask association.

Saved a batch summary containing detection and association statistics.

Dataset and Configuration

Camera images: Stage_1/Images/

LiDAR scans: data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/

Calibration files: calibration/

Segmentation model: YOLOv8x-seg

Target class: Car (class ID 2)

Confidence threshold: 0.25

Frames processed: 20

Output Structure

Stage_3/
├── README.md
├── Task1_yolo_car_segmentation.py
├── Task2_visualize_car_masks.py
├── Task3_validate_car_masks.py
├── Task4_associate_lidar_with_masks.py
├── Task5_visualize_car_point_clouds.py
├── Task6_batch_process_frames.py
└── results/
    ├── YOLOv8_Segmentation/
    │   └── <frame_id>/
    │       ├── masks/
    │       ├── detections.csv
    │       └── <frame_id>_yolo_cars.png
    ├── Mask_Validation/
    │   ├── <frame_id>_mask_validation.csv
    │   └── all_frames_mask_validation.csv
    ├── LiDAR_Mask_Association/
    │   └── <frame_id>/
    │       ├── car_point_clouds/
    │       ├── association_summary.csv
    │       └── <frame_id>_association_overlay.png
    └── batch_processing_summary.csv

Results

Processed 20 camera frames with matching LiDAR scans.

Generated car detections and individual segmentation masks.

Validated the saved masks and recorded their properties in CSV reports.

Associated projected LiDAR points with detected car masks.

Exported individual car point clouds and batch-level statistics.

The number of detections and associated LiDAR points varies between frames. Some frames may contain no detected cars or very few associated LiDAR points.

Limitations

Segmentation masks may include background pixels or overlapping objects.

LiDAR-mask association is based on 2D pixel overlap and does not guarantee that every associated point belongs to the detected vehicle.

Distant or partially occluded vehicles may have sparse LiDAR coverage.

The quality of projected points depends on the camera–LiDAR calibration and the segmentation masks.

Next Stage

Stage 4: Camera–LiDAR Fusion

The next stage will build on these outputs to combine camera-based detections with their associated 3D LiDAR information and generate fused visualizations.