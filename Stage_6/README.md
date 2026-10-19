# Stage 6 — Quantitative Evaluation and Validation

## 1. Overview

Stage 6 focuses on the quantitative evaluation, geometric validation, and performance analysis of the LiDAR–camera sensor fusion pipeline developed in the previous stages.

The project associates LiDAR point clouds with 2D vehicle segmentation masks and evaluates the resulting 3D point-cloud assignments against available ground-truth 3D bounding-box annotations.

The primary objectives of Stage 6 are to:

- Audit the available predictions and ground-truth annotations.
- Establish one-to-one associations between predicted LiDAR clouds and ground-truth boxes.
- Evaluate geometric consistency using point containment and spatial-distance measurements.
- Identify potentially unreliable associations.
- Inspect representative examples visually.
- Investigate the geometric topology of ground-truth cuboids.
- Calculate precision, recall, and F1-score across all 20 selected frames.

**Important:** The results are exploratory evaluation metrics. The ground-truth annotations do not contain class labels, the interpretation of the 3D box corner ordering has not been independently verified for every annotation, and the available annotations may include objects other than vehicles.

---

## 2. Project Context

The preceding stages established the data and processing pipeline used for the evaluation.

| Stage | Main purpose |
|---|---|
| Stage 1 | Initial project setup and data inspection |
| Stage 2 | Preparation of the camera and LiDAR data and calibration information |
| Stage 3 | LiDAR point-cloud association with YOLO instance segmentation masks |
| Stage 4 | Visualization and analysis of the fused camera–LiDAR results |
| Stage 5 | Ground-truth validation, geometric evaluation, and diagnostic visualization |
| Stage 6 | Quantitative evaluation, one-to-one matching, error analysis, and performance metrics |

The Stage 6 analysis uses the following inputs:

- Predicted LiDAR point-cloud assignments from Stage 3.
- Ground-truth 3D bounding-box JSON files.
- Geometric evaluation results from Stage 5.
- The global one-to-one matching results generated during Stage 6.

### Dataset summary

| Property | Value |
|---|---:|
| Selected frames | 20 |
| Predicted LiDAR cloud records | 82 |
| Frames containing predicted clouds | 19 |
| Frames without predicted clouds | 1 |
| Ground-truth records across the selected frames | 1,009 |
| Ground-truth annotation format | JSON |
| Ground-truth coordinate representation | Camera 0 coordinates |
| LiDAR point-cloud coordinates | LiDAR coordinates |

The frame without an associated predicted cloud is frame `0000000570`.

The number of ground-truth records varies substantially between frames. In particular, frames 2033, 2098, and 2449 contain 186, 186, and 314 records, respectively. These counts are retained in the evaluation, but the records cannot all be assumed to represent vehicles because class labels are unavailable.

---

## 3. Stage 6 Workflow

The evaluation is organized into seven main tasks.

| Task | Description | Main outcome |
|---|---|---|
| Task 1 | Data audit and matching analysis | Audited predictions and one-to-one associations |
| Task 2 | Preliminary performance metrics | Initial precision, recall, and F1 estimates |
| Task 3 | Geometric evaluation | Point-containment and distance measurements |
| Task 4 | Reliability and error analysis | Potentially problematic matches identified |
| Task 5 | Visual inspection | Diagnostic plots for representative cases |
| Task 6 | Ground-truth cuboid topology analysis | Preliminary investigation of cuboid corner connectivity |
| Task 7 | Full-frame performance evaluation | Final exploratory precision, recall, and F1 results |

---

# 4. Task 1 — Data Audit and One-to-One Matching

## 4.1 Task 1.1 — Prediction and Matching Audit

The first task examined the consistency of the prediction and matching datasets.

The audit identified:

- 82 predicted LiDAR cloud records.
- 8,533 candidate cloud-to-ground-truth association records.
- Predictions distributed across 19 of the 20 selected frames.
- Six frame/ground-truth combinations that were selected by multiple clouds in the initial unconstrained best-candidate evaluation.

The unconstrained evaluation allowed multiple predicted clouds to select the same ground-truth record. Consequently, those initial associations could not be interpreted as unique detection matches.

This motivated the introduction of a one-to-one matching strategy.

## 4.2 Task 1.2 — Ground-Truth Annotation Audit

The ground-truth JSON files were inspected to understand their structure.

Each annotation record contains:

- `index`: the ground-truth record identifier.
- `corners_cam0`: the eight 3D cuboid corners in camera 0 coordinates.

A total of 1,009 ground-truth records were found across the selected frames. Their corner arrays were checked for the expected `(8, 3)` structure and finite numerical values.

No class labels were available in the records. Therefore, the evaluation cannot reliably distinguish vehicle boxes from other annotated object types based on the JSON data alone.

Frames 2033, 2098, and 2449 contain unusually large numbers of ground-truth records. These annotations are included in the reported metrics, but they may substantially influence the calculated recall.

## 4.3 Task 1.3 — Candidate Association Analysis

The candidate association results were examined using point containment and the number of candidate ground-truth boxes associated with each cloud.

Key observations:

- 66 of 82 clouds had at least one LiDAR point inside their selected best candidate.
- 16 clouds had zero points inside their selected best candidate.
- The mean best-candidate containment ratio was 46.84%.
- The median containment ratio was 57.53%.
- The standard deviation was 28.68%.
- The minimum and maximum ratios were 0% and 91.98%, respectively.

The number of candidate cloud-to-box pairs meeting different containment thresholds was also examined.

| Minimum containment ratio | Candidate pairs |
|---|---:|
| At least 1% | 182 |
| At least 10% | 114 |
| At least 30% | 80 |
| At least 50% | 60 |
| At least 70% | 17 |

These values describe candidate associations rather than unique final matches.

## 4.4 Task 1.4 — Greedy One-to-One Matching

A greedy matching strategy was applied using a minimum containment threshold of 30% and a requirement that at least one point lie inside the candidate box.

The procedure selected associations while preventing a predicted cloud or ground-truth record from being assigned more than once.

Results:

- 59 provisional matches.
- 23 unmatched predicted clouds.
- Zero duplicated ground-truth assignments in the resulting one-to-one matches.
- Mean containment ratio of matched clouds: 62.41%.
- Median containment ratio: 64.04%.
- Minimum matched containment ratio: 31.58%.
- Maximum matched containment ratio: 91.98%.

This provided a more consistent basis for evaluating prediction-to-ground-truth associations.

## 4.5 Task 1.5 — Global Optimal One-to-One Matching

A global matching approach using SciPy's `linear_sum_assignment` was implemented.

The matching objective prioritized the number of valid matches and then the containment ratio. Candidate associations below the specified 30% threshold were not accepted as valid matches.

The global matching produced the same overall result as the greedy strategy:

| Metric | Result |
|---|---:|
| Predicted clouds | 82 |
| One-to-one matches | 59 |
| Unmatched predictions | 23 |
| Duplicate assignments | 0 |
| Mean containment ratio of matches | 62.41% |
| Median containment ratio of matches | 64.04% |

The agreement between the two methods indicates that both produced the same selected matching result for this dataset and configuration. It does not, by itself, establish that the selected associations are physically correct.

**Main output:**

`Stage_6/results/Task1/Task1_5/global_one_to_one_matches.csv`

---

# 5. Task 2 — Preliminary Performance Metrics

Task 2 calculated an initial set of performance indicators using the available predictions, ground-truth annotations, and matching results.

The preliminary calculation reported:

- 82 predicted clouds.
- 59 matched predictions.
- 23 unmatched predictions.
- 1,009 ground-truth records.
- 59 matched ground-truth records.

The preliminary precision, recall, and F1 estimates were approximately 71.95%, 5.85%, and 10.82%, respectively.

These preliminary figures were generated using an earlier metric calculation and are **not the final Task 7 results**. Task 7 subsequently applied an explicit 50% containment threshold to classify matched predictions and included all 82 predicted clouds in the prediction denominator.

The Task 2 figures are retained as historical exploratory results rather than being used as the final reported performance.

---

# 6. Task 3 — Geometric Evaluation

Task 3 evaluated the spatial relationship between each one-to-one matched LiDAR cloud and its selected ground-truth bounding box.

The analysis calculated point-containment statistics and a representative-point-to-box-center distance.

## 6.1 Point-Containment Ratio

The containment ratio is calculated as:

\[
R_{\mathrm{inside}} =
\frac{N_{\mathrm{inside}}}
{N_{\mathrm{inside}}+N_{\mathrm{outside}}}
\times 100\%
\]

where:

- \(N_{\mathrm{inside}}\) is the number of evaluated points inside the selected ground-truth box.
- \(N_{\mathrm{outside}}\) is the number of evaluated points outside the box.

A high containment ratio indicates that a large fraction of the evaluated cloud lies inside the selected box. A low ratio indicates limited containment.

Containment alone does not establish that a match is correct. The result also depends on the box geometry, coordinate transformations, point-cloud association, and annotation quality.

## 6.2 Distance Measurement

The evaluation includes `center_distance_m`, which measures the distance between a representative point of the cloud and the center of the selected box.

This is a center-distance measurement, not a point-to-surface distance or a 3D intersection-over-union (3D IoU) measurement.

For example, frame 250, `car_02`, was associated with ground-truth record 176:

| Measurement | Value |
|---|---:|
| Finite cloud points | 29 |
| Points inside | 13 |
| Points outside | 16 |
| Containment ratio | 44.83% |
| Representative-point-to-box-center distance | 19.009 m |

This example was retained for further diagnostic inspection because of its relatively low containment and large center distance.

## 6.3 Task 3 Outputs

The following files were generated:

- `Stage_6/results/Task3/per_match_geometric_metrics.csv`
- `Stage_6/results/Task3/per_frame_geometric_metrics.csv`
- `Stage_6/results/Task3/overall_geometric_metrics.csv`

Diagnostic plots were also generated to support analysis of the geometric results.

---

# 7. Task 4 — Reliability and Error Analysis

Task 4 examined the matched clouds for potential geometric or data-quality issues.

Eight of the 59 matches were flagged for review. The flag categories overlapped, so eight flagged matches does not mean eight independent error categories.

The diagnostic criteria included:

- Small point-cloud size.
- Low containment ratio.
- Large representative-point-to-box-center distance.

The flagged records were intended to identify cases requiring additional scrutiny, not automatically classify them as incorrect.

Sparse point clouds may have insufficient geometric coverage to support reliable evaluation. Low containment may arise from incorrect associations, partial visibility, annotation differences, calibration issues, or assumptions about the cuboid geometry.

Similarly, a large center distance is a diagnostic indicator rather than proof of a calibration error.

The Task 4 findings should therefore be interpreted as a quality-control review of the current matching results.

---

# 8. Task 5 — Visual Inspection

Task 5 generated diagnostic visualizations to examine selected LiDAR-to-ground-truth associations.

The visual inspection concentrated on representative examples, including frame 250, `car_02`.

The plots showed apparent spatial misalignment and cuboid-edge inconsistencies in the selected example. This motivated further investigation of the assumed ground-truth corner ordering.

However, a single example cannot establish that the calibration matrix is incorrect. Other possible causes include:

- An incorrect interpretation of cuboid corner ordering.
- Differences between coordinate systems.
- Incomplete or sparse point clouds.
- Incorrect cloud-to-box association.
- Annotation conventions that have not yet been independently confirmed.

The visual inspection was therefore treated as a diagnostic step rather than a definitive calibration validation.

---

# 9. Task 6 — Ground-Truth Cuboid Topology Analysis

Task 6 investigated the assumed connectivity of the eight ground-truth cuboid corners.

For the inspected example, frame 250, ground-truth record 176, a set of candidate cuboid edges was examined:

- `(0,1)`, `(0,2)`, `(1,3)`, `(2,3)`
- `(4,5)`, `(4,6)`, `(5,7)`, `(6,7)`
- `(0,5)`, `(1,4)`, `(2,7)`, `(3,6)`

The measured edge lengths suggested a plausible cuboid structure for this example. However, the corner ordering has not been independently validated against the annotation specification for all ground-truth records.

A nearest-neighbour approach was also investigated but was not considered reliable enough to establish the complete cuboid topology.

Consequently:

1. The inspected edge arrangement is a working hypothesis.
2. The same corner ordering must not be assumed to be verified for every annotation.
3. The geometric containment metrics remain provisional until the annotation convention and coordinate transformations are independently validated.

This limitation is particularly important when interpreting the final precision, recall, and F1-score.

---

# 10. Task 7 — Precision, Recall, and F1 Across All Frames

Task 7 calculated the final exploratory performance metrics across all 20 selected frames.

Unlike the earlier preliminary metric calculation, Task 7 explicitly retained all 82 predicted clouds, including the 23 clouds that were not assigned a valid one-to-one ground-truth match.

## 10.1 Classification Method

A 50% containment threshold was used to classify matched predictions.

A matched prediction was classified as a true positive when:

\[
R_{\mathrm{inside}} \geq 50\%
\]

The classification rules were:

| Condition | Classification |
|---|---|
| One-to-one matched cloud with containment ratio ≥ 50% | True Positive (TP) |
| Unmatched predicted cloud | False Positive (FP) |
| Matched cloud with containment ratio < 50% | False Positive (FP) |
| Ground-truth record not covered by a true-positive match | False Negative (FN) |

A matched cloud that falls below the 50% threshold is counted as a false positive, and its ground-truth record remains uncovered by a true positive and is consequently counted as a false negative.

This follows the implemented classification procedure. It is an exploratory evaluation rule rather than a fully validated object-detection benchmark protocol.

## 10.2 Performance Metrics

Precision measures the proportion of predictions classified as true positives:

\[
\mathrm{Precision} =
\frac{TP}{TP+FP}
\]

Recall measures the proportion of ground-truth records covered by true-positive matches:

\[
\mathrm{Recall} =
\frac{TP}{TP+FN}
\]

The F1-score is the harmonic mean of precision and recall:

\[
F_1 =
2\cdot
\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
\]

All three metrics are reported as percentages.

## 10.3 Final Overall Results

The final Task 7 script produced the following results:

| Metric | Final value |
|---|---:|
| Frames evaluated | 20 |
| Ground-truth records | 1,009 |
| Predicted LiDAR clouds | 82 |
| Matched predictions | 59 |
| Unmatched predictions | 23 |
| True positives (TP) | 50 |
| False positives (FP) | 32 |
| False negatives (FN) | 959 |
| Precision | 60.98% |
| Recall | 4.96% |
| F1-score | 9.17% |

### Precision calculation

The 82 predicted clouds resulted in 50 true positives and 32 false positives:

\[
\mathrm{Precision}
=\frac{50}{50+32}\times100
=60.98\%
\]

This means that approximately 61% of the predicted clouds were classified as true positives under the selected matching and containment rules.

### Recall calculation

The evaluation contained 1,009 ground-truth records, of which 50 were covered by true-positive matches:

\[
\mathrm{Recall}
=\frac{50}{50+959}\times100
=4.96\%
\]

The low recall indicates that only a small fraction of the ground-truth records were covered by true-positive matches under the current evaluation procedure.

This result is strongly influenced by the number of ground-truth records in the dataset, the limited number of predicted clouds, and the absence of class labels in the annotation files.

### F1-score calculation

Using the unrounded precision and recall values:

\[
F_1 =
2\cdot\frac{0.609756\cdot0.049554}
{0.609756+0.049554}
\approx0.0917
\]

Therefore, the final F1-score is approximately **9.17%**.

The F1-score is low because the recall is low, even though precision is considerably higher.

## 10.4 Per-Frame Performance

The following table summarizes the results for each evaluated frame.

| Frame | GT records | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 31 | 7 | 6 | 1 | 25 | 85.71% | 19.35% | 31.58% |
| 250 | 31 | 4 | 3 | 1 | 28 | 75.00% | 9.68% | 17.14% |
| 360 | 31 | 4 | 2 | 2 | 29 | 50.00% | 6.45% | 11.43% |
| 570 | 9 | 0 | 0 | 0 | 9 | 0.00% | 0.00% | 0.00% |
| 644 | 10 | 2 | 2 | 0 | 8 | 100.00% | 20.00% | 33.33% |
| 729 | 10 | 1 | 1 | 0 | 9 | 100.00% | 10.00% | 18.18% |
| 947 | 12 | 5 | 5 | 0 | 7 | 100.00% | 41.67% | 58.82% |
| 1134 | 12 | 3 | 3 | 0 | 9 | 100.00% | 25.00% | 40.00% |
| 1280 | 33 | 2 | 1 | 1 | 32 | 50.00% | 3.03% | 5.71% |
| 1338 | 21 | 3 | 2 | 1 | 19 | 66.67% | 9.52% | 16.67% |
| 1461 | 21 | 8 | 4 | 4 | 17 | 50.00% | 19.05% | 27.59% |
| 1597 | 22 | 4 | 4 | 0 | 18 | 100.00% | 18.18% | 30.77% |
| 1791 | 15 | 3 | 0 | 3 | 15 | 0.00% | 0.00% | 0.00% |
| 1866 | 15 | 1 | 0 | 1 | 15 | 0.00% | 0.00% | 0.00% |
| 2033 | 186 | 4 | 3 | 1 | 183 | 75.00% | 1.61% | 3.16% |
| 2098 | 186 | 14 | 9 | 5 | 177 | 64.29% | 4.84% | 9.00% |
| 2449 | 314 | 13 | 3 | 10 | 311 | 23.08% | 0.96% | 1.83% |
| 2717 | 17 | 2 | 0 | 2 | 17 | 0.00% | 0.00% | 0.00% |
| 2903 | 17 | 1 | 1 | 0 | 16 | 100.00% | 5.88% | 11.11% |
| 2939 | 16 | 1 | 1 | 0 | 15 | 100.00% | 6.25% | 11.76% |

The per-frame results show substantial variation. Frames 947 and 1134 achieved relatively high F1-scores under the current evaluation rule, whereas frames 2033, 2098, and 2449 showed very low recall due to their large ground-truth annotation counts.

Frame 570 contained nine ground-truth records but no predicted clouds, resulting in zero precision, recall, and F1 under the script's zero-denominator convention.

These per-frame metrics are useful for identifying differences between frames, but they should not be interpreted as independent proof of sensor calibration or detection quality.

## 10.5 Interpretation of the Results

The final results indicate a difference between prediction precision and ground-truth coverage.

**Precision of 60.98%:** A majority of the predicted clouds satisfied the implemented true-positive rule. However, this does not prove that the associated objects have been identified correctly in every case.

**Recall of 4.96%:** Only 50 of the 1,009 ground-truth records were covered by true-positive matches. The very large ground-truth counts in several frames contribute substantially to this low value.

**F1-score of 9.17%:** The harmonic mean is low because recall is low. The F1-score therefore reflects the limited coverage achieved by the current set of predictions under the specified matching procedure.

These values should not be interpreted as a direct measurement of the performance of YOLO alone. The evaluation concerns the resulting LiDAR cloud assignments and their geometric associations with the available ground-truth records.

The results also do not establish whether the dominant source of error is segmentation, point-cloud association, annotation interpretation, calibration, or another component of the pipeline.

---

# 11. Output Files

Task 7 generated three principal CSV files.

```text
Stage_6/
├── README.md
├── Task7_precision_recall_f1_all_frames.py
└── results/
    ├── Task1/
    │   └── Task1_5/
    │       └── global_one_to_one_matches.csv
    ├── Task3/
    │   ├── per_match_geometric_metrics.csv
    │   ├── per_frame_geometric_metrics.csv
    │   └── overall_geometric_metrics.csv
    └── Task7/
        ├── per_frame_precision_recall_f1.csv
        ├── overall_precision_recall_f1.csv
        └── per_detection_classification.csv
```

The Task 3 directory also contains the diagnostic plots generated during geometric evaluation.

### Output descriptions

**`global_one_to_one_matches.csv`**

Contains the globally selected one-to-one prediction-to-ground-truth associations.

**`per_match_geometric_metrics.csv`**

Contains geometric evaluation measurements for individual matched clouds.

**`per_frame_geometric_metrics.csv`**

Summarizes geometric evaluation measurements at frame level.

**`overall_geometric_metrics.csv`**

Contains aggregate geometric evaluation results.

**`per_frame_precision_recall_f1.csv`**

Contains TP, FP, FN, precision, recall, and F1-score for each evaluated frame.

**`overall_precision_recall_f1.csv`**

Contains the final aggregate metrics across all 20 frames.

**`per_detection_classification.csv`**

Contains prediction-level matching information and the classification assigned to each predicted cloud.

---

# 12. Reproducing Task 7

Run the script from the parent `Computer_Vision_Projects` directory using the project's Python virtual environment.

```powershell
python .\Lidar_Camera_Fusion\Stage_6\Task7_precision_recall_f1_all_frames.py
```

The script reads the Stage 5 prediction CSV and the Stage 6 global matching CSV, loads the corresponding ground-truth JSON files, computes the metrics, and writes the three Task 7 CSV outputs.

The evaluation configuration uses:

- All 20 selected frames.
- All 82 prediction records.
- The 59 global one-to-one matches.
- A 50% point-containment threshold for true-positive classification.

---

# 13. Limitations

The following limitations are important when interpreting the results.

### 13.1 Missing ground-truth class labels

The ground-truth JSON files contain indices and cuboid corners but no class labels. The evaluation therefore counts all ground-truth records, even though they cannot all be confirmed as vehicle annotations.

### 13.2 Uneven annotation counts

Frames 2033, 2098, and 2449 contain unusually large numbers of annotations. Their counts strongly affect the overall recall because the evaluation includes every ground-truth record in the denominator.

### 13.3 Unverified cuboid corner ordering

The assumed cuboid topology was examined for a representative example but has not been independently validated for every ground-truth annotation.

### 13.4 Coordinate-system and geometric assumptions

The ground-truth corners are stored in camera coordinates, while the associated LiDAR clouds are stored in LiDAR coordinates. Geometric comparisons are valid only when the relevant points and boxes are expressed in a consistent coordinate system using the appropriate transformations.

The current results should not be treated as definitive physical calibration validation.

### 13.5 Containment is not 3D IoU

The point-containment ratio measures the fraction of evaluated cloud points inside a box. It is not equivalent to 3D intersection over union, nor does it measure the overlap between two 3D bounding boxes.

### 13.6 Unmatched predictions and thresholding

All unmatched predicted clouds are counted as false positives, while matched predictions below the 50% containment threshold are classified as false positives. This is the explicit rule used by Task 7, but it does not account for every possible detection-evaluation convention, such as excluding detections that cannot be evaluated because of occlusion.

### 13.7 Exploratory metric status

The final metrics describe the current project implementation and its assumptions. They should not be compared directly with published detection benchmarks unless the annotation definitions, coordinate transformations, matching protocol, and evaluation rules are made equivalent.

---

# 14. Conclusion

Stage 6 established an evaluation workflow for examining LiDAR point-cloud assignments against the available ground-truth 3D bounding boxes.

The work included dataset auditing, candidate association analysis, greedy and global one-to-one matching, geometric evaluation, reliability diagnostics, visual inspection, cuboid topology investigation, and full-frame precision, recall, and F1 calculation.

The final Task 7 evaluation covered 20 frames, 82 predicted clouds, and 1,009 ground-truth records. It produced 50 true positives, 32 false positives, and 959 false negatives, resulting in a precision of 60.98%, recall of 4.96%, and F1-score of 9.17%.

These metrics provide a quantitative baseline for the current pipeline. The low recall and F1-score highlight limited ground-truth coverage under the implemented evaluation rules, while the geometric and annotation limitations mean the results should be treated as exploratory rather than definitive benchmark performance.

Overall, Stage 6 provides a structured record of the current system's evaluation, its observed limitations, and the evidence available for interpreting its performance.
