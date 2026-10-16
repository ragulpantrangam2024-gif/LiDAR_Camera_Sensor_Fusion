# Stage 5 — Ground-Truth Validation and Quantitative Evaluation

## Overview

Stage 5 evaluates the LiDAR–camera fusion pipeline developed in the previous stages. It validates ground-truth 3D bounding-box annotations, investigates coordinate transformations, evaluates associated LiDAR point clouds against ground-truth boxes, calculates quantitative diagnostic metrics, and generates visual comparisons.

The evaluation uses 20 selected frames from the KITTI-360 dataset.

## Objectives

- Validate the structure and numerical validity of ground-truth 3D bounding boxes.
- Inspect the LiDAR-to-camera coordinate transformation.
- Investigate ground-truth box geometry and point-in-box containment.
- Evaluate fused LiDAR point clouds across all selected frames.
- Calculate quantitative cloud-containment and quality metrics.
- Visualize high-overlap and low-overlap examples to identify potential mismatches.

## Tasks Completed

### Task 1 — Ground-Truth 3D Bounding Box Validation

**Script:** `Task1_inspect_ground_truth_boxes.py`

Validated the ground-truth bounding-box annotations stored in JSON files.

Checks included:
- Availability of ground-truth annotations.
- Eight 3D corners per bounding box.
- Correct corner-array shape of `(8, 3)`.
- Finite numerical coordinates.

**Results:**
- Frames inspected: 20
- Bounding boxes inspected: 1,009
- Valid bounding boxes: 1,009
- Invalid bounding boxes: 0

**Output:**
- `results/ground_truth_validation.csv`

### Task 2A — LiDAR-to-Camera Coordinate Transform Verification

**Script:** `Task2_check_coordinate_transform.py`

Inspected the calibration matrix in `calibration/calib_cam_to_velo.txt` and examined the transformation between LiDAR and camera coordinate systems.

Checks included:
- Loading the 3 × 4 calibration matrix.
- Constructing the homogeneous 4 × 4 transformation matrix.
- Computing its inverse.
- Testing the mathematical round-trip transformation.

The round-trip test verifies the matrix inversion, but does not independently establish the physical correctness of the calibration convention.

### Task 2B — Ground-Truth 3D Bounding Box Geometry Inspection

**Script:** `Task2_inspect_box_geometry.py`

Inspected ground-truth box corner coordinates, centers, and edge lengths.

The task investigated the corner ordering needed for point-in-box calculations. The edge convention was inferred from inspected examples and has not been independently verified for every annotation.

### Task 2C — Point-in-Box Diagnostic and Debugging

Investigated whether transformed LiDAR points lie inside ground-truth 3D bounding boxes.

The diagnostic:
- Transforms LiDAR points into camera coordinates.
- Examines candidate ground-truth boxes.
- Counts points inside each candidate.
- Calculates containment ratios and center distances.
- Reports candidate boxes for further investigation.

These results are diagnostic because they depend on the calibration convention, box-corner ordering, and candidate-selection method.

### Task 3 — Batch Point-in-Box Evaluation Across All Frames

**Script:** `Task3_batch_point_in_box_evaluation.py`

Evaluated the associated Stage 3 LiDAR point clouds against ground-truth boxes across all selected frames.

The script selects a best candidate box for each cloud based on point containment, with additional criteria used to rank candidates.

**Results:**
- LiDAR cloud files evaluated: 82
- Cloud–box candidate records: 8,533
- Candidate evaluations marked `OK`: 77
- Sparse-cloud records: 5
- Clouds with at least one point inside the selected box: 66 of 82
- Mean best-candidate inside ratio: 46.84%

**Output files:**
- `results/Task3/per_car_evaluation.csv`
- `results/Task3/all_box_candidates.csv`
- `results/Task3/evaluation_warnings.csv`

A selected candidate is not necessarily the correct ground-truth object. Multiple clouds may select the same ground-truth box.

### Task 4 — Quantitative Fusion Evaluation and Metrics

**Script:** `Task4_quantitative_fusion_metrics.py`

Calculated quantitative diagnostic statistics from the Task 3 evaluation results.

#### Overall results

| Metric | Result |
|---|---:|
| Cloud records evaluated | 82 |
| Non-empty clouds with valid ratios | 82 |
| Sparse clouds | 5 |
| Total finite point records across clouds | 40,933 |
| Points inside selected best candidates | 25,949 |
| Points outside selected best candidates | 14,984 |
| Mean inside ratio | 46.84% |
| Median inside ratio | 57.53% |
| Minimum inside ratio | 0.00% |
| Maximum inside ratio | 91.98% |
| Clouds with at least one point inside | 66 |
| Clouds with zero points inside | 16 |
| Clouds below 30% inside | 22 |
| Clouds with at least 70% inside | 17 |

The point totals are summed across cloud records and should not be interpreted as unique LiDAR points.

#### Cloud quality categories

| Category | Count |
|---|---:|
| High overlap | 17 |
| Moderate overlap | 43 |
| Low overlap | 17 |
| Sparse cloud | 5 |

The sparse-cloud category is treated separately from the ordinary overlap categories.

**Output files:**
- `results/Task4/overall_metrics.csv`
- `results/Task4/per_frame_metrics.csv`
- `results/Task4/cloud_quality_analysis.csv`
- `results/Task4/low_overlap_clouds.csv`

**Important:** The inside ratio is a containment metric. It is not precision, recall, F1-score, or 3D Intersection over Union (IoU).

### Task 5 — Visual Evaluation Against Ground-Truth 3D Boxes

**Script:** `Task5_visualize_ground_truth_comparison.py`

Generated 3D visualizations for high-overlap and low-overlap point clouds.

The script:
- Selects up to three high-overlap and three low-overlap examples.
- Transforms LiDAR points into camera coordinates.
- Draws the selected ground-truth 3D box.
- Saves plots for visual inspection.
- Exports the selected examples to CSV.

**Examples inspected:**
- Frame 1461, `car_01`: 85.39% inside ratio.
- Frame 2098, `car_05`: 91.97% inside ratio.
- Frame 2098, `car_07`: 86.26% inside ratio.
- Frame 100, `car_06`: 0% inside ratio.
- Frame 360, `car_01`: 0% inside ratio.
- Frame 360, `car_04`: 0% inside ratio.

The visual comparison shows promising high-overlap examples as well as cases where the selected box and point cloud do not align. These cases motivate further investigation rather than proving a calibration error.

**Output directory:**
- `results/Task5/plots/`
- `results/Task5/selected_visual_examples.csv`

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- JSON-based ground-truth annotation processing
- LiDAR point-cloud processing
- 3D coordinate transformations
- Quantitative evaluation and visualization

## Stage 5 Directory Structure

```text
Stage_5/
├── README.md
├── Task1_inspect_ground_truth_boxes.py
├── Task2_check_coordinate_transform.py
├── Task2_inspect_box_geometry.py
├── Task3_batch_point_in_box_evaluation.py
├── Task4_quantitative_fusion_metrics.py
├── Task5_visualize_ground_truth_comparison.py
└── results/
    ├── ground_truth_validation.csv
    ├── Task3/
    │   ├── per_car_evaluation.csv
    │   ├── all_box_candidates.csv
    │   └── evaluation_warnings.csv
    ├── Task4/
    │   ├── overall_metrics.csv
    │   ├── per_frame_metrics.csv
    │   ├── cloud_quality_analysis.csv
    │   └── low_overlap_clouds.csv
    └── Task5/
        ├── selected_visual_examples.csv
        └── plots/
```

The tree above lists the main deliverables; additional diagnostic scripts and output files may also exist.

## Limitations

- A mathematically valid inverse transformation does not independently validate the physical calibration convention.
- The ground-truth box corner ordering is based on inspected examples and needs broader verification.
- Best-candidate selection does not guarantee correct object identity or one-to-one matching.
- Point containment can be affected by sparse clouds, segmentation errors, outliers, and coordinate-system inconsistencies.
- The total number of point records across clouds is not necessarily the number of unique LiDAR points.
- Precision, recall, F1-score, and validated 3D IoU accuracy have not yet been established.

## Future Work — Stage 6

Stage 6 will establish a reliable ground-truth matching protocol and evaluate detection and association performance using appropriate metrics, including:

- Precision
- Recall
- F1-score
- 3D IoU, where valid predicted 3D boxes are available
- Additional geometric error and association-quality measurements

## Conclusion

Stage 5 establishes a ground-truth validation and diagnostic evaluation workflow for the LiDAR–camera fusion pipeline. It processes 82 associated point clouds, calculates quantitative containment metrics, and generates visual comparisons across high- and low-overlap cases.

The results help identify promising associations and potential failure cases. They provide a foundation for the more rigorous performance evaluation planned for Stage 6.
