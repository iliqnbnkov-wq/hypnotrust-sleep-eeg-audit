"""
Calibration module for HypnoTrust v3b.

Fits a multiclass Platt-style calibrator (LogisticRegression on log-probabilities)
using out-of-fold (OOF) probabilities from development subjects 0-8.
Applies the calibrator to Subject 9 locked final evaluation.

Class order is always locked to STAGE_ORDER from constants.py.
Subject 9 is forbidden from all fitting steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from hypnotrust.constants import (
    STAGE_ORDER,
    EXCLUDED_TEST_SUBJECTS,
)
from hypnotrust.uncertainty import (
    validate_probability_matrix,
    maximum_confidence,
    predicted_class_indices,
    predictive_entropy,
    multiclass_log_loss,
    multiclass_brier_score,
    expected_calibration_error,
    selective_performance_curve,
)


_STAGE_TO_INT: dict[str, int] = {stage: i for i, stage in enumerate(STAGE_ORDER)}
_INT_TO_STAGE: dict[int, str] = {i: stage for i, stage in enumerate(STAGE_ORDER)}

OOF_PROB_COLUMNS: tuple[str, ...] = (
    "Uncalibrated_Probability_Wake",
    "Uncalibrated_Probability_N1",
    "Uncalibrated_Probability_N2",
    "Uncalibrated_Probability_N3",
    "Uncalibrated_Probability_REM",
)

CALIBRATED_PROB_COLUMNS: tuple[str, ...] = (
    "Calibrated_Probability_Wake",
    "Calibrated_Probability_N1",
    "Calibrated_Probability_N2",
    "Calibrated_Probability_N3",
    "Calibrated_Probability_REM",
)

_EXPECTED_CALIBRATOR_CLASSES: np.ndarray = np.arange(len(STAGE_ORDER))


def _require_no_subject_9(df: pd.DataFrame, context: str) -> None:
    """Raise ValueError if Subject 9 is present in df."""
    forbidden = set(EXCLUDED_TEST_SUBJECTS)
    present = set(df["Subject"].unique()) & forbidden
    if present:
        raise ValueError(
            f"{context}: Subject 9 must never be used for fitting. "
            f"Found subjects: {sorted(present)}"
        )


def _reorder_proba_to_stage_order(
    proba: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    """Reorder predict_proba() output columns to match STAGE_ORDER."""
    model_class_list = list(model_classes)
    missing = [stage for stage in STAGE_ORDER if stage not in model_class_list]
    if missing:
        raise ValueError(
            f"Model classes missing stages required by STAGE_ORDER: {missing}. "
            f"model.classes_ = {model_class_list}"
        )

    column_indices = [model_class_list.index(stage) for stage in STAGE_ORDER]
    return proba[:, column_indices]


def _encode_labels_to_int(y: Sequence[str] | pd.Series | np.ndarray) -> np.ndarray:
    """Encode stage label strings to integers according to STAGE_ORDER."""
    y_array = np.asarray(y)
    unknown = set(y_array) - set(_STAGE_TO_INT)
    if unknown:
        raise ValueError(
            f"Unknown stage labels not in STAGE_ORDER: {sorted(unknown)}"
        )
    return np.array([_STAGE_TO_INT[label] for label in y_array])


def _clip_log_proba(
    probs: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return log(clip(probs, epsilon, 1.0))."""
    return np.log(np.clip(probs, epsilon, 1.0))


def _require_valid_calibrator(calibrator: LogisticRegression) -> None:
    """Raise ValueError if calibrator.classes_ is not [0, 1, 2, 3, 4]."""
    if not hasattr(calibrator, "classes_") or not np.array_equal(
        calibrator.classes_,
        _EXPECTED_CALIBRATOR_CLASSES,
    ):
        raise ValueError(
            f"calibrator.classes_ must be {list(_EXPECTED_CALIBRATOR_CLASSES)} "
            f"(integer-encoded STAGE_ORDER). "
            f"Got: {getattr(calibrator, 'classes_', 'missing')}. "
            f"Use fit_calibrator() to produce a compatible calibrator."
        )


def generate_oof_probabilities(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    group_column: str,
    model_config: dict,
    cv_config: dict,
) -> pd.DataFrame:
    """Generate out-of-fold probabilities on development subjects.

    Each OOF probability comes from a model that has never seen the
    corresponding subject during training.
    """
    _require_no_subject_9(df, "generate_oof_probabilities")

    df = df.reset_index(drop=True)
    X = df[feature_columns].to_numpy()
    y = df[label_column].to_numpy()
    groups = df[group_column].to_numpy()

    number_of_stages = len(STAGE_ORDER)
    oof_probs = np.full((len(df), number_of_stages), np.nan)
    oof_predictions = np.full(len(df), "", dtype=object)

    splitter = StratifiedGroupKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )

    for train_index, validation_index in splitter.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            min_samples_leaf=model_config["min_samples_leaf"],
            class_weight=model_config["class_weight"],
            random_state=model_config["random_state"],
            n_jobs=-1,
        )

        model.fit(X[train_index], y[train_index])

        raw_probabilities = model.predict_proba(X[validation_index])
        ordered_probabilities = _reorder_proba_to_stage_order(
            raw_probabilities,
            model.classes_,
        )
        validate_probability_matrix(ordered_probabilities)

        oof_probs[validation_index] = ordered_probabilities

        predicted_indices = predicted_class_indices(ordered_probabilities)
        oof_predictions[validation_index] = np.array(
            [_INT_TO_STAGE[index] for index in predicted_indices]
        )

    if np.any(np.isnan(oof_probs)):
        raise ValueError(
            "OOF probabilities contain NaN. Some epochs were not covered by any fold. "
            "Check that n_splits <= number of unique subjects."
        )

    result = df[["Subject", "Recording", "Epoch", label_column]].copy()
    result = result.rename(columns={label_column: "True_Stage"})
    result["Predicted_Stage_Uncalibrated"] = oof_predictions

    for column, stage in zip(OOF_PROB_COLUMNS, STAGE_ORDER):
        stage_index = _STAGE_TO_INT[stage]
        result[column] = oof_probs[:, stage_index]

    return result.reset_index(drop=True)


def fit_calibrator(
    oof_df: pd.DataFrame,
    prob_columns: list[str],
    label_column: str,
    epsilon: float = 1e-7,
) -> LogisticRegression:
    """Fit a multiclass Platt-style calibrator on OOF probabilities."""
    _require_no_subject_9(oof_df, "fit_calibrator")

    if len(prob_columns) != len(STAGE_ORDER):
        raise ValueError(
            f"prob_columns must contain exactly {len(STAGE_ORDER)} columns, "
            f"got {len(prob_columns)}: {prob_columns}"
        )

    probabilities = oof_df[prob_columns].to_numpy()
    validate_probability_matrix(probabilities)

    X_calibration = _clip_log_proba(probabilities, epsilon)
    y_calibration = _encode_labels_to_int(oof_df[label_column])

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    calibrator.fit(X_calibration, y_calibration)

    _require_valid_calibrator(calibrator)

    return calibrator


def apply_calibrator(
    calibrator: LogisticRegression,
    probs: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Apply a fitted calibrator to a probability matrix."""
    _require_valid_calibrator(calibrator)

    probabilities = np.asarray(probs)

    if probabilities.ndim != 2 or probabilities.shape[1] != len(STAGE_ORDER):
        raise ValueError(
            f"probs must have shape [n_epochs, {len(STAGE_ORDER)}], "
            f"got {probabilities.shape}"
        )

    validate_probability_matrix(probabilities)

    X_calibration = _clip_log_proba(probabilities, epsilon)
    calibrated = calibrator.predict_proba(X_calibration)

    row_sums = calibrated.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-5))[0]
        raise ValueError(
            f"Calibrated probabilities do not sum to 1.0 (atol=1e-5) "
            f"for {len(bad_rows)} rows. First bad row index: {bad_rows[0]}, "
            f"sum = {row_sums[bad_rows[0]]:.8f}"
        )

    return calibrated


def compute_calibration_comparison(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    label_names: Sequence[str],
    n_bins: int = 10,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> dict:
    """Compare uncalibrated and calibrated probability outputs."""
    if len(label_names) != len(STAGE_ORDER):
        raise ValueError(
            f"label_names must have exactly {len(STAGE_ORDER)} entries, "
            f"got {len(label_names)}: {label_names}"
        )

    y_true = np.asarray(y_true)

    if np.any(y_true < 0) or np.any(y_true >= len(label_names)):
        bad_values = np.unique(
            y_true[(y_true < 0) | (y_true >= len(label_names))]
        )
        raise ValueError(
            f"y_true contains values out of range [0, {len(label_names)}): "
            f"{bad_values}. Expected integer labels 0-{len(label_names) - 1} "
            f"corresponding to {label_names}."
        )

    validate_probability_matrix(uncalibrated_probs)
    validate_probability_matrix(calibrated_probs)

    def _compute_metrics(probabilities: np.ndarray) -> dict:
        confidence = maximum_confidence(probabilities)
        predictions = predicted_class_indices(probabilities)
        entropy = predictive_entropy(probabilities, normalize=True)

        correct_mask = predictions == y_true
        incorrect_mask = ~correct_mask

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            cohen_kappa_score,
        )

        y_true_string = np.array([label_names[index] for index in y_true])
        prediction_string = np.array([label_names[index] for index in predictions])

        return {
            "ece": expected_calibration_error(
                y_true,
                predictions,
                confidence,
                number_of_bins=n_bins,
            ),
            "log_loss": multiclass_log_loss(
                y_true_string,
                probabilities,
                label_names,
            ),
            "brier_score": multiclass_brier_score(
                y_true_string,
                probabilities,
                label_names,
            ),
            "mean_confidence_correct": (
                float(confidence[correct_mask].mean())
                if correct_mask.any()
                else float("nan")
            ),
            "mean_confidence_incorrect": (
                float(confidence[incorrect_mask].mean())
                if incorrect_mask.any()
                else float("nan")
            ),
            "mean_entropy_correct": (
                float(entropy[correct_mask].mean())
                if correct_mask.any()
                else float("nan")
            ),
            "mean_entropy_incorrect": (
                float(entropy[incorrect_mask].mean())
                if incorrect_mask.any()
                else float("nan")
            ),
            "accuracy": float(accuracy_score(y_true_string, prediction_string)),
            "macro_f1": float(
                f1_score(
                    y_true_string,
                    prediction_string,
                    average="macro",
                    zero_division=0,
                )
            ),
            "kappa": float(cohen_kappa_score(y_true_string, prediction_string)),
        }

    def _selective_to_list(probabilities: np.ndarray) -> list[dict]:
        confidence = maximum_confidence(probabilities)
        predictions = predicted_class_indices(probabilities)

        y_true_string = np.array([label_names[index] for index in y_true])
        prediction_string = np.array([label_names[index] for index in predictions])

        curve_dataframe = selective_performance_curve(
            y_true_string,
            prediction_string,
            confidence,
            coverages=coverages,
        )

        return curve_dataframe.to_dict(orient="records")

    uncalibrated_predictions = predicted_class_indices(uncalibrated_probs)
    calibrated_predictions = predicted_class_indices(calibrated_probs)

    return {
        "uncalibrated": _compute_metrics(uncalibrated_probs),
        "calibrated": _compute_metrics(calibrated_probs),
        "n_labels_changed": int(
            np.sum(uncalibrated_predictions != calibrated_predictions)
        ),
        "selective_prediction": {
            "uncalibrated": _selective_to_list(uncalibrated_probs),
            "calibrated": _selective_to_list(calibrated_probs),
        },
    }
