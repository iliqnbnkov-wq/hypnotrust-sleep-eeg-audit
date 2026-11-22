from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def _require_labels(y_true: list[str] | np.ndarray | None, context: str) -> None:
    """
    Guard against metric computation without verified ground-truth labels.
    """

    if y_true is None:
        raise ValueError(
            f"{context}: ground-truth labels are required. "
            "Do not compute accuracy, F1, kappa, confusion matrix, ECE, "
            "Brier score, log loss, or selective prediction metrics without "
            "a verified hypnogram."
        )


def _validate_metric_inputs(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    stage_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and normalize metric inputs.
    """

    _require_labels(y_true, "metrics input validation")

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    probabilities_array = np.asarray(probabilities, dtype=float)

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be a 1D list or array.")

    if y_pred_array.ndim != 1:
        raise ValueError("y_pred must be a 1D list or array.")

    if probabilities_array.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            f"y_true length ({len(y_true_array)}) must match "
            f"y_pred length ({len(y_pred_array)})."
        )

    if probabilities_array.shape[0] != len(y_true_array):
        raise ValueError(
            f"probabilities rows ({probabilities_array.shape[0]}) must match "
            f"y_true length ({len(y_true_array)})."
        )

    if probabilities_array.shape[1] != len(stage_names):
        raise ValueError(
            f"probabilities columns ({probabilities_array.shape[1]}) must match "
            f"stage_names length ({len(stage_names)})."
        )

    known_stages = set(stage_names)
    unknown_true = sorted(set(y_true_array) - known_stages)
    unknown_pred = sorted(set(y_pred_array) - known_stages)

    if unknown_true:
        raise ValueError(f"Unknown labels found in y_true: {unknown_true}")

    if unknown_pred:
        raise ValueError(f"Unknown labels found in y_pred: {unknown_pred}")

    return y_true_array, y_pred_array, probabilities_array


def compute_expected_calibration_error(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    stage_names: list[str],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE compares confidence against empirical correctness.
    Therefore it requires ground-truth labels.
    """

    y_true_array, y_pred_array, probabilities_array = _validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        stage_names=stage_names,
    )

    confidences = np.max(probabilities_array, axis=1)
    correct = y_true_array == y_pred_array

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if not np.any(in_bin):
            continue

        bin_accuracy = float(np.mean(correct[in_bin]))
        bin_confidence = float(np.mean(confidences[in_bin]))
        bin_weight = float(np.mean(in_bin))

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_brier_score_multiclass(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    stage_names: list[str],
) -> float:
    """
    Compute multiclass Brier score.

    This compares predicted probabilities against one-hot encoded true labels.
    """

    y_true_array, y_pred_array, probabilities_array = _validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        stage_names=stage_names,
    )

    stage_to_index = {stage: index for index, stage in enumerate(stage_names)}
    y_one_hot = np.zeros_like(probabilities_array, dtype=float)

    for row_index, label in enumerate(y_true_array):
        y_one_hot[row_index, stage_to_index[str(label)]] = 1.0

    return float(np.mean(np.sum((probabilities_array - y_one_hot) ** 2, axis=1)))


def _log_loss_in_stage_order(
    y_true_array: np.ndarray,
    probabilities_array: np.ndarray,
    stage_names: list[str],
) -> float:
    """
    Multiclass log loss with probability columns given in ``stage_names`` order.

    scikit-learn's ``log_loss`` binarizes ``y_true`` in *sorted* label order
    and expects probability columns in that same order. Passing columns in
    clinical order (Wake, N1, N2, N3, REM) with ``labels=stage_names`` silently
    mismatches them, so the columns are reordered to sorted label order first.
    """

    order = sorted(range(len(stage_names)), key=lambda i: stage_names[i])
    sorted_labels = [stage_names[i] for i in order]

    return float(
        log_loss(
            y_true_array,
            probabilities_array[:, order],
            labels=sorted_labels,
        )
    )


def compute_metrics_if_available(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    stage_names: list[str],
) -> dict:
    """
    Compute metrics only when verified ground-truth labels are available.

    This function intentionally raises ValueError when y_true is None.
    The app layer must check for labels before calling it.
    """

    y_true_array, y_pred_array, probabilities_array = _validate_metric_inputs(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        stage_names=stage_names,
    )

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "macro_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=stage_names,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=stage_names,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true_array, y_pred_array)),
        "ece": compute_expected_calibration_error(
            y_true_array,
            y_pred_array,
            probabilities_array,
            stage_names,
        ),
        "brier_score": compute_brier_score_multiclass(
            y_true_array,
            y_pred_array,
            probabilities_array,
            stage_names,
        ),
        "log_loss": _log_loss_in_stage_order(
            y_true_array,
            probabilities_array,
            stage_names,
        ),
    }


def build_confusion_matrix_df(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    stage_names: list[str],
) -> pd.DataFrame:
    """
    Build confusion matrix DataFrame.

    Requires ground-truth labels.
    """

    _require_labels(y_true, "confusion matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=stage_names,
    )

    return pd.DataFrame(
        cm,
        index=[f"true_{stage}" for stage in stage_names],
        columns=[f"pred_{stage}" for stage in stage_names],
    )


def build_per_stage_metrics_df(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    stage_names: list[str],
) -> pd.DataFrame:
    """
    Build per-stage precision, recall, F1 and support table.

    Requires ground-truth labels.
    """

    _require_labels(y_true, "per-stage metrics")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=stage_names,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "stage": stage_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_top_error_pairs_df(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
) -> pd.DataFrame:
    """
    Build table of most common true/predicted error pairs.

    Requires ground-truth labels.
    """

    _require_labels(y_true, "top error pairs")

    df = pd.DataFrame(
        {
            "true_stage": list(y_true),
            "predicted_stage": list(y_pred),
        }
    )

    errors = df[df["true_stage"] != df["predicted_stage"]]

    if errors.empty:
        return pd.DataFrame(
            columns=["true_stage", "predicted_stage", "count"]
        )

    return (
        errors.groupby(["true_stage", "predicted_stage"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def build_selective_prediction_curve_df(
    y_true: list[str] | np.ndarray | None,
    y_pred: list[str] | np.ndarray,
    confidences: list[float] | np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> pd.DataFrame:
    """
    Build selective prediction curve.

    For each coverage level, keeps the highest-confidence epochs and computes
    accuracy on that retained subset. Requires ground-truth labels.
    """

    _require_labels(y_true, "selective prediction curve")

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    confidences_array = np.asarray(confidences, dtype=float)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(confidences_array) != len(y_true_array):
        raise ValueError("confidences length must match y_true length.")

    n_total = len(y_true_array)
    order = np.argsort(-confidences_array)

    rows = []

    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"coverage must be in (0, 1], got {coverage}")

        n_keep = max(1, int(round(coverage * n_total)))
        kept_indices = order[:n_keep]

        retained_accuracy = accuracy_score(
            y_true_array[kept_indices],
            y_pred_array[kept_indices],
        )

        rows.append(
            {
                "coverage": float(coverage),
                "n_retained": int(n_keep),
                "n_reviewed": int(n_total - n_keep),
                "retained_accuracy": float(retained_accuracy),
                "confidence_threshold": float(
                    np.min(confidences_array[kept_indices])
                ),
            }
        )

    return pd.DataFrame(rows)