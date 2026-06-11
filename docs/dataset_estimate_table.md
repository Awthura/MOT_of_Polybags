---
geometry: margin=2.5cm
fontsize: 12pt
---

# Annotation Scale Estimate
### From 86 manually annotated frames, extrapolated to the full 1,157-frame dataset

\

| Class | Count in 86 frames | Avg per frame | Total (1,157 frames) |
|:---|---:|---:|---:|
| Pink | 169 | 1.97 | 2,274 |
| Blue | 90 | 1.05 | 1,211 |
| Yellow | 256 | 2.98 | 3,444 |
| Grey | 86 | 1.00 | 1,157 |
| Green | 235 | 2.73 | 3,162 |
| Red | 83 | 0.97 | 1,117 |
| **Total** | **919** | **10.69** | **12,365** |

\

At ~10.7 instances per frame across 1,157 frames, fully manual annotation
would require drawing and labelling approximately **12,365 oriented bounding boxes** —
justifying the use of an automated labelling pipeline.

\

## Manual Correction Estimate
### Precision 88.8% · Recall 93.0% applied to N = 12,365 estimated instances

\

| Category | Formula | Count |
|:---|:---|---:|
| True Positives (TP) — correct auto-labels | Recall × N = 0.930 × 12,365 | 11,499 |
| False Negatives (FN) — missed instances | (1 − Recall) × N = 0.070 × 12,365 | 866 |
| False Positives (FP) — spurious detections | TP × (1/Precision − 1) = 11,499 × 0.126 | 1,450 |
| **Total manual interventions (FN + FP)** | | **2,316** |

\

| Metric | Value |
|:---|---:|
| True Positives (no correction required) | 11,499 · 93.0% of N |
| False Negatives (instances to be added) | 866 · 7.0% of N |
| False Positives (instances to be deleted) | 1,450 · 11.2% of auto-detections |
| Total manual interventions | 2,316 · 18.7% of N |
| **Annotation effort reduction vs. fully manual** | **81.3%** |
