# HypnoTrust V5d — Frozen External Evaluation Report

## Status

V5d frozen external evaluation completed and finalized.

This is an experiment-layer external stability check. It does not overwrite the locked v1/v2/v3/v3b/v4 results.

## Split

- Training subjects: ['SC400', 'SC401', 'SC402', 'SC403', 'SC404', 'SC405', 'SC406', 'SC407', 'SC408']
- Historical locked test subject: ['SC409']
- V5 external holdout subjects: ['SC410', 'SC411', 'SC412', 'SC413', 'SC414', 'SC415', 'SC416', 'SC417', 'SC418', 'SC419']

## Protection checks

- Subject 9 used for training: NO
- Subject 9 used for tuning: NO
- External holdout subjects used for training: NO
- External holdout subjects used for tuning: NO
- Locked results modified: NO

## Feature/model setup

- Base features: 13 EEG/EOG engineered features
- Temporal context: ±2 epochs
- Temporal features: 65
- Model: RandomForestClassifier
- n_estimators: 300
- class_weight: balanced_subsample
- random_state: 42

## Historical locked Subject 9 evaluation

| Metric | Value |
|---|---:|
| `n_epochs` | 1132 |
| `accuracy` | 0.8127 |
| `balanced_accuracy` | 0.7453 |
| `macro_f1` | 0.7333 |
| `weighted_f1` | 0.8108 |
| `cohen_kappa` | 0.7120 |
| `log_loss` | 0.6942 |
| `brier_score` | 0.3064 |
| `ece` | 0.0930 |
| `error_rate` | 0.1873 |

## Subject 9 reference comparison

- Locked Temporal v2 Subject 9 accuracy reference: 0.8313
- V5d reconstructed Subject 9 accuracy: 0.8127
- Absolute delta: 0.0186
- Reference status: PASS_CLOSE_TO_LOCKED_TEMPORAL_V2_REFERENCE

## Historical Subject 9 selective prediction at 80% coverage

- Retained accuracy: 0.8620
- Review queue error rate: 0.3850
- Review epochs: 226

## V5 external holdout pooled evaluation

| Metric | Value |
|---|---:|
| `n_epochs` | 10713 |
| `accuracy` | 0.7752 |
| `balanced_accuracy` | 0.6675 |
| `macro_f1` | 0.6947 |
| `weighted_f1` | 0.7739 |
| `cohen_kappa` | 0.6692 |
| `log_loss` | 0.6151 |
| `brier_score` | 0.3161 |
| `ece` | 0.0634 |
| `error_rate` | 0.2248 |

## V5 external holdout selective prediction at 80% coverage

- Retained accuracy: 0.8559
- Review queue error rate: 0.5474
- Review epochs: 2143

## V5 external holdout per-subject metrics

| Subject | Epochs | Accuracy | Balanced acc | Macro F1 | Kappa | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `SC410` | 1104 | 0.8496 | 0.6584 | 0.6228 | 0.7513 | 0.1711 |
| `SC411` | 928 | 0.6961 | 0.6689 | 0.6234 | 0.5564 | 0.1056 |
| `SC412` | 1052 | 0.7766 | 0.6151 | 0.6354 | 0.6739 | 0.0827 |
| `SC413` | 1028 | 0.8385 | 0.8004 | 0.7834 | 0.7668 | 0.1027 |
| `SC414` | 1004 | 0.8048 | 0.6906 | 0.7229 | 0.7184 | 0.0516 |
| `SC415` | 952 | 0.8498 | 0.7259 | 0.7292 | 0.7970 | 0.1115 |
| `SC416` | 1144 | 0.7386 | 0.6740 | 0.6742 | 0.6326 | 0.0576 |
| `SC417` | 1002 | 0.6707 | 0.5068 | 0.5153 | 0.5369 | 0.0497 |
| `SC418` | 964 | 0.8071 | 0.7099 | 0.7049 | 0.7276 | 0.0583 |
| `SC419` | 1535 | 0.7362 | 0.5413 | 0.5890 | 0.5274 | 0.0487 |

## External holdout summary across subjects

| Metric | Mean | Std | Min | Max | N subjects |
|---|---:|---:|---:|---:|---:|
| `n_epochs` | 1071.3000 | 176.1414 | 928.0000 | 1535.0000 | 10 |
| `accuracy` | 0.7768 | 0.0641 | 0.6707 | 0.8498 | 10 |
| `balanced_accuracy` | 0.6591 | 0.0865 | 0.5068 | 0.8004 | 10 |
| `macro_f1` | 0.6600 | 0.0785 | 0.5153 | 0.7834 | 10 |
| `weighted_f1` | 0.7691 | 0.0718 | 0.6470 | 0.8561 | 10 |
| `cohen_kappa` | 0.6688 | 0.1001 | 0.5274 | 0.7970 | 10 |
| `log_loss` | 0.6157 | 0.1604 | 0.4435 | 0.9350 | 10 |
| `brier_score` | 0.3156 | 0.0744 | 0.2243 | 0.4391 | 10 |
| `ece` | 0.0839 | 0.0394 | 0.0487 | 0.1711 | 10 |
| `mean_confidence_correct` | 0.7647 | 0.0465 | 0.6616 | 0.8188 | 10 |
| `mean_confidence_incorrect` | 0.5191 | 0.0450 | 0.4564 | 0.5803 | 10 |
| `error_rate` | 0.2232 | 0.0641 | 0.1502 | 0.3293 | 10 |

## Top error pairs — historical Subject 9

| True | Predicted | Errors | Fraction of errors | Mean confidence |
|---|---|---:|---:|---:|
| REM | N2 | 81 | 0.3821 | 0.7038 |
| N3 | N2 | 56 | 0.2642 | 0.7688 |
| N2 | REM | 27 | 0.1274 | 0.5033 |
| N2 | N1 | 15 | 0.0708 | 0.4016 |
| N2 | Wake | 10 | 0.0472 | 0.4000 |
| Wake | N1 | 5 | 0.0236 | 0.4387 |
| N1 | Wake | 5 | 0.0236 | 0.4113 |
| REM | N1 | 5 | 0.0236 | 0.3580 |
| N1 | N2 | 3 | 0.0142 | 0.4767 |
| N3 | Wake | 3 | 0.0142 | 0.4189 |

## Top error pairs — V5 external holdout pooled

| True | Predicted | Errors | Fraction of errors | Mean confidence |
|---|---|---:|---:|---:|
| REM | N2 | 502 | 0.2085 | 0.5697 |
| N3 | N2 | 354 | 0.1470 | 0.6150 |
| Wake | N2 | 340 | 0.1412 | 0.5007 |
| N2 | REM | 331 | 0.1375 | 0.4656 |
| N1 | N2 | 202 | 0.0839 | 0.5780 |
| Wake | N1 | 169 | 0.0702 | 0.4509 |
| REM | N1 | 101 | 0.0419 | 0.3730 |
| Wake | REM | 98 | 0.0407 | 0.4771 |
| N1 | REM | 94 | 0.0390 | 0.5108 |
| N2 | N1 | 73 | 0.0303 | 0.3994 |

## Interpretation

The V5d reconstruction is close to the locked Temporal v2 Subject 9 reference. The Subject 9 accuracy delta is 0.0186, producing the status `PASS_CLOSE_TO_LOCKED_TEMPORAL_V2_REFERENCE`.

On the newly added external holdout subjects, pooled accuracy is 0.7752, macro F1 is 0.6947, and Cohen's kappa is 0.6692. This indicates a measurable cross-subject generalization drop compared with the historical Subject 9 result.

This is scientifically useful: it shows that the original locked result was not enough to characterize broader cross-subject stability.

The external holdout evaluation should be presented as an additional robustness/stability analysis, not as a replacement for the locked v2/v3/v3b/v4 hierarchy.

## Guardrails

- V5d is not a clinical validation.
- V5d does not establish clinical decision-making readiness.
- V5d does not make HypnoTrust a diagnostic system.
- The external holdout subjects were not used for training or tuning.
- Subject 9 remains protected from training, tuning, calibration, alpha selection, and review-policy design.

## Files used/created

- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_subject_metrics.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_external_summary.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_selective_prediction.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_epoch_predictions.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_error_pairs.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_per_stage_metrics.csv`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_confusion_matrices.json`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_eval_config.json`
- `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/V5D_FROZEN_EXTERNAL_EVALUATION_REPORT.md`

## V5d final result

PASS
