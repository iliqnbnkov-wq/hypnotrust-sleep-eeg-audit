"""
Integration test for EDF evaluation with SC4001 (Sleep-EDF).

This test verifies that the EDF pipeline produces the expected metrics
when evaluated against the verified SC4001 hypnogram.

Reference values (from frozen V5d evaluation):
- Accuracy: 0.9702
- Balanced Accuracy: 0.8340
- Macro F1: 0.8407
- Cohen's Kappa: 0.9276
- Sleep-only Accuracy: 0.8821
- Sleep-only Macro F1: 0.8039

Confusion matrix (rows=true, columns=predicted):
          Wake  N1   N2   N3  REM
Wake      1995   1    0    0    1
N1           7  19    4    0   28
N2           1   0  220   20    9
N3           0   0    8  212    0
REM          0   0    0    0  125

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_sc4001_edf_evaluation():
    """
    Integration test: Run full EDF pipeline on SC4001 and verify metrics.

    This test:
    1. Loads SC4001E0-PSG.edf and SC4001EC-Hypnogram.edf
    2. Runs the full feature extraction pipeline
    3. Runs the temporal v2 model prediction
    4. Evaluates against verified hypnogram
    5. Verifies metrics match reference values within tolerance
    """
    # Paths to SC4001 files
    psg_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001E0-PSG.edf"
    hypnogram_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001EC-Hypnogram.edf"

    # Skip test if files are not available
    if not psg_path.exists():
        pytest.skip(f"PSG file not found: {psg_path}")

    if not hypnogram_path.exists():
        pytest.skip(f"Hypnogram file not found: {hypnogram_path}")

    print(f"Loading PSG from: {psg_path}")
    print(f"Loading hypnogram from: {hypnogram_path}")

    # Load PSG
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    print(f"PSG loaded: {raw.n_times} samples, {raw.info['sfreq']} Hz")

    # Load hypnogram
    annotations = mne.read_annotations(hypnogram_path)
    print(f"Hypnogram loaded: {len(annotations)} annotations")

    # Run pipeline
    from core.config import get_default_pipeline_config
    from core.pipeline import run_full_pipeline
    from core.edf_evaluation import evaluate_edf_predictions

    config = get_default_pipeline_config()

    print("Running full EDF pipeline...")
    result = run_full_pipeline(
        raw=raw,
        annotations=annotations,
        config=config,
        subject=4001,  # Use integer for validation
        recording=0,
    )

    print(f"Pipeline complete: {len(result.predicted_stages)} epochs")

    # Evaluate
    print("Computing evaluation metrics...")
    evaluation_result = evaluate_edf_predictions(
        y_true=result.labels,
        y_pred=result.predicted_stages,
        probabilities=result.probabilities,
        stage_names=result.stage_names,
        raw_duration_sec=float(raw.n_times / raw.info["sfreq"]),
        annotations_provided=True,
        model_name="Temporal v2 RandomForest",
        uncertainty_policy=config.uncertainty_policy,
        epoch_length_sec=config.epoch_length_sec,
        recording_identity="SC4001E0",
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total epochs: {evaluation_result.n_total_epochs}")
    print(f"Valid epochs: {evaluation_result.n_valid_epochs}")
    print(f"Excluded epochs: {evaluation_result.n_excluded_epochs}")
    print()

    metrics = evaluation_result.metrics
    print("Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print()

    if evaluation_result.wake_only_metrics:
        print("Wake metrics:")
        print(f"  Wake Recall: {evaluation_result.wake_only_metrics['wake_recall']:.4f}")
        print(f"  Epochs: {evaluation_result.wake_only_metrics['wake_n_epochs']}")
        print()

    if evaluation_result.always_wake_baseline:
        print("Always-Wake Baseline:")
        print(f"  Baseline Accuracy: {evaluation_result.always_wake_baseline['always_wake_baseline_accuracy']:.4f}")
        print(f"  Total Epochs: {evaluation_result.always_wake_baseline['total_epochs']}")
        print()

    if evaluation_result.sleep_only_metrics:
        print("Sleep-only metrics:")
        print(f"  Accuracy: {evaluation_result.sleep_only_metrics['sleep_only_accuracy']:.4f}")
        print(f"  Macro F1: {evaluation_result.sleep_only_metrics['sleep_only_macro_f1']:.4f}")
        print(f"  Epochs: {evaluation_result.sleep_only_metrics['sleep_only_n_epochs']}")
        print()

    print("Confusion matrix:")
    print(evaluation_result.confusion_matrix.to_string())
    print()

    # Verify against reference values
    print("=" * 60)
    print("VERIFICATION AGAINST REFERENCE VALUES")
    print("=" * 60)

    TOLERANCE = 0.01  # 1% tolerance

    reference_accuracy = 0.9702
    reference_balanced_accuracy = 0.8340
    reference_macro_f1 = 0.8407
    reference_kappa = 0.9276
    reference_sleep_only_accuracy = 0.8821
    reference_sleep_only_macro_f1 = 0.8039
    reference_wake_recall = 1995 / 1997  # Expected: 0.9990
    reference_always_wake_baseline = 1997 / 2650  # Expected: 0.7536

    accuracy_match = abs(metrics['accuracy'] - reference_accuracy) < TOLERANCE
    balanced_accuracy_match = abs(metrics['balanced_accuracy'] - reference_balanced_accuracy) < TOLERANCE
    macro_f1_match = abs(metrics['macro_f1'] - reference_macro_f1) < TOLERANCE
    kappa_match = abs(metrics['cohen_kappa'] - reference_kappa) < TOLERANCE

    print(f"Accuracy: {metrics['accuracy']:.4f} vs {reference_accuracy:.4f} - {'[MATCH]' if accuracy_match else '[MISMATCH]'}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} vs {reference_balanced_accuracy:.4f} - {'[MATCH]' if balanced_accuracy_match else '[MISMATCH]'}")
    print(f"Macro F1: {metrics['macro_f1']:.4f} vs {reference_macro_f1:.4f} - {'[MATCH]' if macro_f1_match else '[MISMATCH]'}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f} vs {reference_kappa:.4f} - {'[MATCH]' if kappa_match else '[MISMATCH]'}")

    if evaluation_result.sleep_only_metrics:
        sleep_acc_match = abs(evaluation_result.sleep_only_metrics['sleep_only_accuracy'] - reference_sleep_only_accuracy) < TOLERANCE
        sleep_f1_match = abs(evaluation_result.sleep_only_metrics['sleep_only_macro_f1'] - reference_sleep_only_macro_f1) < TOLERANCE
        print(f"Sleep-only Accuracy: {evaluation_result.sleep_only_metrics['sleep_only_accuracy']:.4f} vs {reference_sleep_only_accuracy:.4f} - {'[MATCH]' if sleep_acc_match else '[MISMATCH]'}")
        print(f"Sleep-only Macro F1: {evaluation_result.sleep_only_metrics['sleep_only_macro_f1']:.4f} vs {reference_sleep_only_macro_f1:.4f} - {'[MATCH]' if sleep_f1_match else '[MISMATCH]'}")

    if evaluation_result.wake_only_metrics:
        wake_recall_match = abs(evaluation_result.wake_only_metrics['wake_recall'] - reference_wake_recall) < TOLERANCE
        print(f"Wake Recall: {evaluation_result.wake_only_metrics['wake_recall']:.4f} vs {reference_wake_recall:.4f} - {'[MATCH]' if wake_recall_match else '[MISMATCH]'}")

    if evaluation_result.always_wake_baseline:
        baseline_match = abs(evaluation_result.always_wake_baseline['always_wake_baseline_accuracy'] - reference_always_wake_baseline) < TOLERANCE
        print(f"Always-Wake Baseline: {evaluation_result.always_wake_baseline['always_wake_baseline_accuracy']:.4f} vs {reference_always_wake_baseline:.4f} - {'[MATCH]' if baseline_match else '[MISMATCH]'}")

    all_match = accuracy_match and balanced_accuracy_match and macro_f1_match and kappa_match

    if all_match:
        print("\n[PASS] All main metrics match reference values within tolerance")
    else:
        print("\n[FAIL] Some metrics do not match reference values")

    return all_match


if __name__ == "__main__":
    success = test_sc4001_edf_evaluation()
    sys.exit(0 if success else 1)
