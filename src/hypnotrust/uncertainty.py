"""
Uncertainty and calibration utilities for HypnoTrust.

The module evaluates predicted class probabilities without containing
dataset-specific or model-training logic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def validate_probability_matrix(
    probabilities: ArrayLike,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Validate and return a multiclass probability matrix."""
    matrix = np.asarray(probabilities, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Probability matrix must be two-dimensional.")

    number_of_samples, number_of_classes = matrix.shape

    if number_of_samples == 0:
        raise ValueError("Probability matrix must contain at least one sample.")

    if number_of_classes < 2:
        raise ValueError("Probability matrix must contain at least two classes.")

    if not np.isfinite(matrix).all():
        raise ValueError("Probability matrix contains NaN or infinite values.")

    if (matrix < -tolerance).any() or (matrix > 1.0 + tolerance).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    row_sums = matrix.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Every probability row must sum to 1.")

    return matrix


def _validate_classes(
    classes: Sequence[object],
    *,
    expected_count: int,
) -> np.ndarray:
    """Validate class labels and return them as an object array."""
    class_array = np.asarray(list(classes), dtype=object)

    if class_array.ndim != 1:
        raise ValueError("Classes must be a one-dimensional sequence.")

    if len(class_array) != expected_count:
        raise ValueError(
            "Number of classes does not match the probability matrix."
        )

    if pd.isna(class_array).any():
        raise ValueError("Class labels must not contain missing values.")

    if len(set(class_array.tolist())) != len(class_array):
        raise ValueError("Class labels must be unique.")

    return class_array


def _validate_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_length: int,
) -> np.ndarray:
    """Validate a one-dimensional vector."""
    vector = np.asarray(values, dtype=object)

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(vector) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values.")

    if pd.isna(vector).any():
        raise ValueError(f"{name} must not contain missing values.")

    return vector


def maximum_confidence(probabilities: ArrayLike) -> np.ndarray:
    """Return the largest predicted probability for every sample."""
    matrix = validate_probability_matrix(probabilities)
    return matrix.max(axis=1)


def predicted_class_indices(probabilities: ArrayLike) -> np.ndarray:
    """Return the most probable class index for every sample."""
    matrix = validate_probability_matrix(probabilities)
    return matrix.argmax(axis=1)


def predictive_entropy(
    probabilities: ArrayLike,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Calculate predictive entropy for every sample."""
    matrix = validate_probability_matrix(probabilities)

    safe_matrix = np.clip(
        matrix,
        np.finfo(float).tiny,
        1.0,
    )

    entropy = -np.sum(
        matrix * np.log(safe_matrix),
        axis=1,
    )

    if normalize:
        entropy = entropy / np.log(matrix.shape[1])

    return entropy


def build_uncertainty_dataframe(
    probabilities: ArrayLike,
    classes: Sequence[object],
) -> pd.DataFrame:
    """Build a per-sample prediction and uncertainty table."""
    matrix = validate_probability_matrix(probabilities)
    class_array = _validate_classes(classes, expected_count=matrix.shape[1])

    predicted_indices = matrix.argmax(axis=1)

    return pd.DataFrame(
        {
            "Predicted_Class": class_array[predicted_indices],
            "Maximum_Confidence": matrix.max(axis=1),
            "Predictive_Entropy": predictive_entropy(matrix, normalize=True),
        }
    )


def multiclass_log_loss(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    classes: Sequence[object],
    *,
    epsilon: float = 1e-15,
) -> float:
    """Calculate multiclass logarithmic loss."""
    matrix = validate_probability_matrix(probabilities)
    class_array = _validate_classes(classes, expected_count=matrix.shape[1])
    true_labels = _validate_vector(
        y_true,
        name="y_true",
        expected_length=matrix.shape[0],
    )

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1.")

    class_to_index = {
        class_label: class_index
        for class_index, class_label in enumerate(class_array)
    }

    unknown_labels = sorted(
        {label for label in true_labels if label not in class_to_index},
        key=str,
    )

    if unknown_labels:
        raise ValueError(
            "y_true contains labels that are not present in classes: "
            f"{unknown_labels}"
        )

    true_indices = np.asarray(
        [class_to_index[label] for label in true_labels],
        dtype=int,
    )

    true_probabilities = matrix[
        np.arange(len(true_labels)),
        true_indices,
    ]

    clipped_probabilities = np.clip(
        true_probabilities,
        epsilon,
        1.0,
    )

    return float(-np.mean(np.log(clipped_probabilities)))


def multiclass_brier_score(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    classes: Sequence[object],
) -> float:
    """Calculate multiclass Brier score."""
    matrix = validate_probability_matrix(probabilities)
    class_array = _validate_classes(classes, expected_count=matrix.shape[1])
    true_labels = _validate_vector(
        y_true,
        name="y_true",
        expected_length=matrix.shape[0],
    )

    class_to_index = {
        class_label: class_index
        for class_index, class_label in enumerate(class_array)
    }

    unknown_labels = sorted(
        {label for label in true_labels if label not in class_to_index},
        key=str,
    )

    if unknown_labels:
        raise ValueError(
            "y_true contains labels that are not present in classes: "
            f"{unknown_labels}"
        )

    true_indices = np.asarray(
        [class_to_index[label] for label in true_labels],
        dtype=int,
    )

    one_hot_targets = np.zeros_like(matrix, dtype=float)
    one_hot_targets[np.arange(len(true_labels)), true_indices] = 1.0

    squared_error = (matrix - one_hot_targets) ** 2

    return float(np.mean(np.sum(squared_error, axis=1)))


def calibration_table(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence: ArrayLike,
    *,
    number_of_bins: int = 10,
) -> pd.DataFrame:
    """Create a confidence-versus-accuracy calibration table."""
    confidence_array = np.asarray(confidence, dtype=float)

    if confidence_array.ndim != 1:
        raise ValueError("confidence must be one-dimensional.")

    number_of_samples = len(confidence_array)

    if number_of_samples == 0:
        raise ValueError("confidence must contain at least one value.")

    true_labels = _validate_vector(
        y_true,
        name="y_true",
        expected_length=number_of_samples,
    )

    predicted_labels = _validate_vector(
        y_pred,
        name="y_pred",
        expected_length=number_of_samples,
    )

    if not np.isfinite(confidence_array).all():
        raise ValueError("confidence contains NaN or infinite values.")

    if (confidence_array < 0.0).any() or (confidence_array > 1.0).any():
        raise ValueError("confidence values must be between 0 and 1.")

    if (
        not isinstance(number_of_bins, int)
        or isinstance(number_of_bins, bool)
        or number_of_bins < 1
    ):
        raise ValueError("number_of_bins must be a positive integer.")

    correctness = (true_labels == predicted_labels).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_indices = np.searchsorted(
        bin_edges[1:-1],
        confidence_array,
        side="right",
    )

    rows: list[dict[str, float | int]] = []

    for bin_index in range(number_of_bins):
        mask = bin_indices == bin_index
        bin_count = int(mask.sum())

        lower_bound = float(bin_edges[bin_index])
        upper_bound = float(bin_edges[bin_index + 1])

        if bin_count == 0:
            mean_confidence = np.nan
            accuracy = np.nan
            calibration_gap = np.nan
        else:
            mean_confidence = float(confidence_array[mask].mean())
            accuracy = float(correctness[mask].mean())
            calibration_gap = abs(accuracy - mean_confidence)

        rows.append(
            {
                "Bin": bin_index + 1,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
                "Count": bin_count,
                "Mean_Confidence": mean_confidence,
                "Accuracy": accuracy,
                "Calibration_Gap": calibration_gap,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence: ArrayLike,
    *,
    number_of_bins: int = 10,
) -> float:
    """Calculate top-label Expected Calibration Error."""
    table = calibration_table(
        y_true,
        y_pred,
        confidence,
        number_of_bins=number_of_bins,
    )

    total_samples = int(table["Count"].sum())
    non_empty_bins = table[table["Count"] > 0]

    weighted_gaps = (
        non_empty_bins["Count"]
        / total_samples
        * non_empty_bins["Calibration_Gap"]
    )

    return float(weighted_gaps.sum())


def selective_performance_curve(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence: ArrayLike,
    *,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> pd.DataFrame:
    """Evaluate performance when low-confidence samples are rejected."""
    confidence_array = np.asarray(confidence, dtype=float)

    if confidence_array.ndim != 1:
        raise ValueError("confidence must be one-dimensional.")

    number_of_samples = len(confidence_array)

    if number_of_samples == 0:
        raise ValueError("confidence must contain at least one value.")

    true_labels = _validate_vector(
        y_true,
        name="y_true",
        expected_length=number_of_samples,
    )

    predicted_labels = _validate_vector(
        y_pred,
        name="y_pred",
        expected_length=number_of_samples,
    )

    if not np.isfinite(confidence_array).all():
        raise ValueError("confidence contains NaN or infinite values.")

    if (confidence_array < 0.0).any() or (confidence_array > 1.0).any():
        raise ValueError("confidence values must be between 0 and 1.")

    coverage_values = [float(coverage) for coverage in coverages]

    if not coverage_values:
        raise ValueError("At least one coverage value is required.")

    if any(coverage <= 0.0 or coverage > 1.0 for coverage in coverage_values):
        raise ValueError(
            "Every coverage must be greater than 0 and less than or equal to 1."
        )

    sorted_indices = np.argsort(
        -confidence_array,
        kind="mergesort",
    )

    rows: list[dict[str, float | int]] = []

    for requested_coverage in coverage_values:
        retained_count = max(
            1,
            int(np.ceil(requested_coverage * number_of_samples)),
        )

        retained_indices = sorted_indices[:retained_count]

        retained_correctness = (
            true_labels[retained_indices]
            == predicted_labels[retained_indices]
        )

        accuracy = float(retained_correctness.mean())
        actual_coverage = retained_count / number_of_samples

        threshold = float(confidence_array[retained_indices].min())

        rows.append(
            {
                "Requested_Coverage": requested_coverage,
                "Actual_Coverage": actual_coverage,
                "Retained_Count": retained_count,
                "Rejected_Count": number_of_samples - retained_count,
                "Confidence_Threshold": threshold,
                "Accuracy": accuracy,
                "Risk": 1.0 - accuracy,
                "Mean_Confidence": float(confidence_array[retained_indices].mean()),
            }
        )

    return pd.DataFrame(rows)
