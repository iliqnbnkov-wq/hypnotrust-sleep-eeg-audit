"""
EDF Evaluation Module for HypnoTrust.

Scientifically rigorous evaluation of model predictions against verified Sleep-EDF hypnogram annotations.

This module implements:
- Label safety: never uses placeholder Stage values as ground truth
- Valid label masking: computes metrics only for epochs with verified annotations
- Wake-only and sleep-only metrics
- Scientific provenance tracking
- CSV and JSON export with research-use disclaimers

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import (
    compute_metrics_if_available,
    build_confusion_matrix_df,
    build_per_stage_metrics_df,
)


STAGE_ORDER = ("Wake", "N1", "N2", "N3", "REM")


@dataclass(frozen=True)
class EDFEvaluationResult:
    """
    Result of EDF evaluation against verified hypnogram annotations.

    Attributes:
        metrics: Dictionary of computed metrics (accuracy, balanced_accuracy, macro_f1, cohen_kappa, etc.)
        confusion_matrix: Confusion matrix DataFrame
        per_stage_metrics: Per-stage precision, recall, F1, support DataFrame
        wake_only_metrics: Metrics computed only on Wake epochs (now Wake Recall)
        sleep_only_metrics: Metrics computed only on sleep stages (N1, N2, N3, REM)
        always_wake_baseline: Always-Wake baseline accuracy for the entire recording
        valid_label_mask: Boolean array indicating which epochs have valid labels
        n_total_epochs: Total number of epochs in the recording
        n_valid_epochs: Number of epochs with valid labels
        n_excluded_epochs: Number of epochs excluded from evaluation (invalid labels)
        provenance: Dictionary tracking scientific provenance information
    """

    metrics: dict[str, float]
    confusion_matrix: pd.DataFrame
    per_stage_metrics: pd.DataFrame
    wake_only_metrics: dict[str, float] | None
    sleep_only_metrics: dict[str, float] | None
    always_wake_baseline: dict[str, float] | None
    valid_label_mask: np.ndarray
    n_total_epochs: int
    n_valid_epochs: int
    n_excluded_epochs: int
    provenance: dict[str, Any]


def validate_annotation_alignment(
    raw_duration_sec: float,
    annotations: list[tuple[float, float, str]] | None,
    epoch_length_sec: int = 30,
) -> tuple[bool, str]:
    """
    Validate that hypnogram annotations align with PSG timing.

    Args:
        raw_duration_sec: Total duration of PSG recording in seconds
        annotations: List of (onset, duration, description) tuples
        epoch_length_sec: Epoch duration in seconds

    Returns:
        (is_valid, warning_message): True if alignment is valid, otherwise False with warning

    Notes:
        - Trailing annotation overhang beyond PSG end is acceptable (common in Sleep-EDF)
        - Annotations must start at or near t=0
        - Annotations must fully cover the PSG recording
    """
    if annotations is None or len(annotations) == 0:
        return True, "No annotations provided"

    # Extract onsets and durations
    onsets = np.array([a[0] for a in annotations])
    durations = np.array([a[1] for a in annotations])

    first_onset = float(onsets.min())
    last_onset = float(onsets.max())
    last_end = float((onsets + durations).max())

    # Check if annotations start at t=0 (within tolerance)
    start_tolerance = epoch_length_sec  # 1 epoch tolerance
    starts_at_zero = abs(first_onset) < start_tolerance

    if not starts_at_zero:
        return (
            False,
            f"Annotations do not start at t=0. First onset: {first_onset:.1f}s "
            f"(tolerance: {start_tolerance}s)"
        )

    # Check if annotations fully cover PSG
    annotation_coverage_end = min(last_end, raw_duration_sec)
    coverage_duration = annotation_coverage_end - first_onset
    coverage_percentage = coverage_duration / raw_duration_sec

    if coverage_percentage < 0.99:  # 99% coverage required
        return (
            False,
            f"Annotations cover only {coverage_percentage * 100:.1f}% of PSG. "
            f"Required: 99%. Coverage: {coverage_duration:.1f}s / {raw_duration_sec:.1f}s"
        )

    # Check for trailing overhang (acceptable)
    trailing_overhang = last_end - raw_duration_sec
    if trailing_overhang > 0:
        # Trailing overhang is acceptable, but note it
        return (
            True,
            f"Annotations have trailing overhang of {trailing_overhang:.1f}s "
            f"({trailing_overhang / 3600:.2f} hours) beyond PSG end. "
            f"This is acceptable for Sleep-EDF format."
        )

    # Perfect alignment
    return True, ""


def build_valid_label_mask(
    labels: list[str | None],
) -> np.ndarray:
    """
    Build boolean mask for valid labels.

    Valid labels are non-None strings in STAGE_ORDER.
    Placeholder labels (None, "Movement time", "Sleep stage ?") are marked as invalid.

    Args:
        labels: List of labels (may contain None for invalid annotations)

    Returns:
        Boolean array where True indicates a valid label
    """
    valid_stages = set(STAGE_ORDER)
    return np.array([label in valid_stages for label in labels], dtype=bool)


def compute_wake_only_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    stage_names: list[str],
) -> dict[str, float]:
    """
    Compute Wake Recall (accuracy on verified Wake epochs).

    This is the recall for the Wake class: correct Wake predictions / total true Wake epochs.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    wake_mask = y_true_array == "Wake"

    if not np.any(wake_mask):
        return {
            "wake_recall": 0.0,
            "wake_n_epochs": 0,
        }

    from sklearn.metrics import accuracy_score

    wake_recall = float(
        accuracy_score(y_true_array[wake_mask], y_pred_array[wake_mask])
    )

    return {
        "wake_recall": wake_recall,
        "wake_n_epochs": int(np.sum(wake_mask)),
    }


def compute_always_wake_baseline(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    stage_names: list[str],
) -> dict[str, float]:
    """
    Compute Always-Wake Baseline Accuracy.

    This is the accuracy if we always predicted "Wake" for all epochs.
    It serves as a trivial baseline for comparison.
    """
    y_true_array = np.asarray(y_true)

    if len(y_true_array) == 0:
        return {
            "always_wake_baseline_accuracy": 0.0,
            "total_epochs": 0,
        }

    # Always predict Wake
    always_wake_predictions = np.array(["Wake"] * len(y_true_array))

    from sklearn.metrics import accuracy_score

    baseline_accuracy = float(
        accuracy_score(y_true_array, always_wake_predictions)
    )

    return {
        "always_wake_baseline_accuracy": baseline_accuracy,
        "total_epochs": len(y_true_array),
    }


def compute_sleep_only_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    stage_names: list[str],
) -> dict[str, float]:
    """
    Compute metrics only on sleep stages (N1, N2, N3, REM).

    This excludes Wake, which is often the dominant class and can inflate overall metrics.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    probabilities_array = np.asarray(probabilities)

    sleep_stages = {"N1", "N2", "N3", "REM"}
    sleep_mask = np.array([label in sleep_stages for label in y_true_array], dtype=bool)

    if not np.any(sleep_mask):
        return {
            "sleep_only_accuracy": 0.0,
            "sleep_only_macro_f1": 0.0,
            "sleep_only_n_epochs": 0,
        }

    from sklearn.metrics import accuracy_score, f1_score

    sleep_only_accuracy = float(
        accuracy_score(y_true_array[sleep_mask], y_pred_array[sleep_mask])
    )

    sleep_only_macro_f1 = float(
        f1_score(
            y_true_array[sleep_mask],
            y_pred_array[sleep_mask],
            labels=["N1", "N2", "N3", "REM"],
            average="macro",
            zero_division=0,
        )
    )

    return {
        "sleep_only_accuracy": sleep_only_accuracy,
        "sleep_only_macro_f1": sleep_only_macro_f1,
        "sleep_only_n_epochs": int(np.sum(sleep_mask)),
    }


def evaluate_edf_predictions(
    y_true: list[str | None],
    y_pred: list[str],
    probabilities: np.ndarray,
    stage_names: list[str],
    raw_duration_sec: float | None = None,
    annotations_provided: bool = False,
    model_name: str = "Temporal v2 RandomForest",
    uncertainty_policy: str = "v3",
    epoch_length_sec: int = 30,
    recording_identity: str = "uploaded",
) -> EDFEvaluationResult:
    """
    Evaluate EDF predictions against verified hypnogram annotations.

    This function implements label safety:
    - Only epochs with valid labels are used for metric computation
    - Placeholder labels (None, Movement time, Sleep stage ?) are excluded
    - Total epoch count is preserved (no index shifting)
    - Metrics are computed only on the valid subset

    Args:
        y_true: Ground-truth labels (may contain None for invalid annotations)
        y_pred: Predicted labels (complete for all epochs)
        probabilities: Predicted probabilities (complete for all epochs)
        stage_names: Ordered list of stage names
        raw_duration_sec: Total PSG duration in seconds (for alignment validation)
        annotations_provided: Whether hypnogram annotations were provided
        model_name: Name of the model used for predictions
        uncertainty_policy: Uncertainty policy version
        epoch_length_sec: Epoch duration in seconds
        recording_identity: Identifier for the recording (e.g., "SC4001E0")

    Returns:
        EDFEvaluationResult with metrics, confusion matrix, and provenance
    """
    n_total_epochs = len(y_pred)

    # Build valid label mask
    valid_label_mask = build_valid_label_mask(y_true)
    n_valid_epochs = int(np.sum(valid_label_mask))
    n_excluded_epochs = n_total_epochs - n_valid_epochs

    provenance = {
        "model_name": model_name,
        "uncertainty_policy": uncertainty_policy,
        "epoch_length_sec": epoch_length_sec,
        "recording_identity": recording_identity,
        "total_epochs": n_total_epochs,
        "valid_epochs": n_valid_epochs,
        "excluded_epochs": n_excluded_epochs,
        "annotations_provided": annotations_provided,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "research_use_only": True,
    }

    # If no valid labels, return result with None metrics
    if n_valid_epochs == 0:
        return EDFEvaluationResult(
            metrics={},
            confusion_matrix=pd.DataFrame(),
            per_stage_metrics=pd.DataFrame(),
            wake_only_metrics=None,
            sleep_only_metrics=None,
            always_wake_baseline=None,
            valid_label_mask=valid_label_mask,
            n_total_epochs=n_total_epochs,
            n_valid_epochs=n_valid_epochs,
            n_excluded_epochs=n_excluded_epochs,
            provenance=provenance,
        )

    # Filter to valid labels for metric computation
    y_true_valid = [label for label, is_valid in zip(y_true, valid_label_mask) if is_valid]
    y_pred_valid = [pred for pred, is_valid in zip(y_pred, valid_label_mask) if is_valid]
    probabilities_valid = probabilities[valid_label_mask]

    # Compute standard metrics
    metrics = compute_metrics_if_available(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        probabilities=probabilities_valid,
        stage_names=stage_names,
    )

    # Build confusion matrix
    confusion_matrix = build_confusion_matrix_df(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        stage_names=stage_names,
    )

    # Build per-stage metrics
    per_stage_metrics = build_per_stage_metrics_df(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        stage_names=stage_names,
    )

    # Compute wake recall and always-wake baseline
    wake_metrics = compute_wake_only_metrics(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        probabilities=probabilities_valid,
        stage_names=stage_names,
    )

    always_wake_baseline = compute_always_wake_baseline(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        probabilities=probabilities_valid,
        stage_names=stage_names,
    )

    # Compute sleep-only metrics
    sleep_only_metrics = compute_sleep_only_metrics(
        y_true=y_true_valid,
        y_pred=y_pred_valid,
        probabilities=probabilities_valid,
        stage_names=stage_names,
    )

    # Add subset counts to provenance
    provenance.update({
        "wake_n_epochs": wake_metrics["wake_n_epochs"],
        "sleep_only_n_epochs": sleep_only_metrics["sleep_only_n_epochs"],
    })

    return EDFEvaluationResult(
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        per_stage_metrics=per_stage_metrics,
        wake_only_metrics=wake_metrics,
        sleep_only_metrics=sleep_only_metrics,
        always_wake_baseline=always_wake_baseline,
        valid_label_mask=valid_label_mask,
        n_total_epochs=n_total_epochs,
        n_valid_epochs=n_valid_epochs,
        n_excluded_epochs=n_excluded_epochs,
        provenance=provenance,
    )


def export_evaluation_csv(
    y_true: list[str | None],
    y_pred: list[str],
    probabilities: np.ndarray,
    confidences: np.ndarray,
    epoch_length_sec: int = 30,
    recording_identity: str = "uploaded",
) -> bytes:
    """
    Export evaluation results to CSV with epoch-by-epoch predictions.

    CSV includes:
    - epoch index
    - time (seconds from start)
    - true label (if available)
    - predicted label
    - confidence
    - probabilities for each stage

    Args:
        y_true: Ground-truth labels (may contain None)
        y_pred: Predicted labels
        probabilities: Predicted probabilities array
        confidences: Max probability per epoch
        epoch_length_sec: Epoch duration in seconds
        recording_identity: Recording identifier

    Returns:
        CSV bytes
    """
    n_epochs = len(y_pred)

    rows = []
    for epoch_idx in range(n_epochs):
        time_sec = epoch_idx * epoch_length_sec
        true_label = y_true[epoch_idx] if epoch_idx < len(y_true) else None
        pred_label = y_pred[epoch_idx]
        confidence = float(confidences[epoch_idx])
        prob_row = probabilities[epoch_idx]

        row = {
            "epoch": epoch_idx,
            "time_sec": time_sec,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
        }

        # Add probability columns
        for stage_idx, stage_name in enumerate(STAGE_ORDER):
            row[f"prob_{stage_name}"] = float(prob_row[stage_idx])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add disclaimer at the top as a comment
    disclaimer = (
        "# HypnoTrust EDF Evaluation Results\n"
        f"# Recording: {recording_identity}\n"
        f"# Epochs: {n_epochs}\n"
        "# RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.\n"
        "# Results must be reviewed by a qualified sleep specialist.\n"
    )

    csv_content = disclaimer + "\n" + df.to_csv(index=False)
    return csv_content.encode("utf-8")


def export_evaluation_json(
    evaluation_result: EDFEvaluationResult,
    recording_identity: str = "uploaded",
) -> bytes:
    """
    Export evaluation results to JSON with metrics and provenance.

    JSON includes:
    - All computed metrics
    - Confusion matrix
    - Per-stage metrics
    - Wake-only and sleep-only metrics
    - Valid/total/excluded epoch counts
    - Scientific provenance information
    - Research-use disclaimer

    Args:
        evaluation_result: EDFEvaluationResult from evaluate_edf_predictions
        recording_identity: Recording identifier

    Returns:
        JSON bytes
    """
    # Convert DataFrames to serializable dicts
    confusion_matrix_dict = evaluation_result.confusion_matrix.to_dict()
    per_stage_metrics_dict = evaluation_result.per_stage_metrics.to_dict()

    export_dict = {
        "recording_identity": recording_identity,
        "metrics": evaluation_result.metrics,
        "wake_metrics": evaluation_result.wake_only_metrics,
        "sleep_only_metrics": evaluation_result.sleep_only_metrics,
        "always_wake_baseline": evaluation_result.always_wake_baseline,
        "confusion_matrix": confusion_matrix_dict,
        "per_stage_metrics": per_stage_metrics_dict,
        "epoch_counts": {
            "total": evaluation_result.n_total_epochs,
            "valid": evaluation_result.n_valid_epochs,
            "excluded": evaluation_result.n_excluded_epochs,
        },
        "provenance": evaluation_result.provenance,
        "disclaimer": (
            "RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE. "
            "Results must be reviewed by a qualified sleep specialist."
        ),
    }

    json_content = json.dumps(export_dict, indent=2)
    return json_content.encode("utf-8")
