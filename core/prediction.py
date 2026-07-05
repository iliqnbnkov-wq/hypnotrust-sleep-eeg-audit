from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from core.types import PipelineConfig


@lru_cache(maxsize=4)
def load_model(model_path: str | Path) -> Any:
    """
    Load a locked model artifact from disk.

    The app must only load existing artifacts.
    It must not train, tune, or overwrite models.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    return joblib.load(path)


def _get_model_classes(model: Any) -> list[str]:
    """
    Extract class labels from a fitted sklearn model or sklearn Pipeline.

    RandomForestClassifier exposes classes_ directly.
    A sklearn Pipeline usually exposes classes_ on the final step.
    """

    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())

        if steps:
            final_step = steps[-1]

            if hasattr(final_step, "classes_"):
                return [str(x) for x in final_step.classes_]

    raise ValueError(
        "Could not find model classes_. "
        "The model must expose class order for safe probability alignment."
    )


def prepare_feature_frame(
    feature_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """
    Validate and reorder feature columns for model inference.

    Extra columns are ignored.
    Missing required columns raise ValueError.
    """

    missing = [col for col in feature_columns if col not in feature_df.columns]

    if missing:
        raise ValueError(
            "Feature DataFrame is missing required model features: "
            f"{missing[:20]}"
            + ("..." if len(missing) > 20 else "")
        )

    ordered = feature_df.loc[:, list(feature_columns)].copy()

    if ordered.isna().any().any():
        nan_columns = ordered.columns[ordered.isna().any()].tolist()
        raise ValueError(
            "Feature DataFrame contains NaN values in required columns: "
            f"{nan_columns[:20]}"
            + ("..." if len(nan_columns) > 20 else "")
        )

    return ordered


def _reorder_probabilities_to_stage_order(
    probabilities: np.ndarray,
    model_classes: list[str],
    stage_order: tuple[str, ...],
) -> np.ndarray:
    """
    Reorder predict_proba columns to match PipelineConfig.stage_order.

    sklearn returns probabilities in model.classes_ order.
    The app uses stage_order consistently for plotting and exports.
    """

    model_class_set = set(model_classes)
    stage_set = set(stage_order)

    if model_class_set != stage_set:
        raise ValueError(
            "Model classes do not match configured stage_order. "
            f"model_classes={model_classes}, stage_order={stage_order}"
        )

    class_to_index = {stage: index for index, stage in enumerate(model_classes)}
    reorder_indices = [class_to_index[stage] for stage in stage_order]

    return probabilities[:, reorder_indices]


def predict_probabilities(
    model: Any,
    feature_df: pd.DataFrame,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Run model.predict_proba on a validated feature DataFrame.

    Returns probabilities in config.stage_order column order.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    model_input = prepare_feature_frame(
        feature_df=feature_df,
        feature_columns=config.feature_columns,
    )

    probabilities = model.predict_proba(model_input)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError(
            f"predict_proba must return a 2D array, got shape {probabilities.shape}"
        )

    if probabilities.shape[0] != len(model_input):
        raise ValueError(
            "predict_proba row count does not match input rows. "
            f"rows={probabilities.shape[0]}, expected={len(model_input)}"
        )

    model_classes = _get_model_classes(model)

    probabilities = _reorder_probabilities_to_stage_order(
        probabilities=probabilities,
        model_classes=model_classes,
        stage_order=config.stage_order,
    )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(
            "Predicted probabilities do not sum to 1. "
            "This indicates an invalid model output."
        )

    return probabilities


def predict_stages_from_probabilities(
    probabilities: np.ndarray,
    stage_order: tuple[str, ...],
) -> list[str]:
    """
    Convert probability matrix to predicted sleep stage labels.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    if probabilities.shape[1] != len(stage_order):
        raise ValueError(
            f"probability columns ({probabilities.shape[1]}) must match "
            f"stage_order length ({len(stage_order)})."
        )

    predicted_indices = np.argmax(probabilities, axis=1)

    return [stage_order[index] for index in predicted_indices]


def run_prediction(
    feature_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[np.ndarray, list[str]]:
    """
    Load the configured model and return probabilities plus predicted stages.

    This is the main prediction entry point used by core.pipeline.
    """

    model = load_model(config.model_path)

    probabilities = predict_probabilities(
        model=model,
        feature_df=feature_df,
        config=config,
    )

    predicted_stages = predict_stages_from_probabilities(
        probabilities=probabilities,
        stage_order=config.stage_order,
    )

    return probabilities, predicted_stages