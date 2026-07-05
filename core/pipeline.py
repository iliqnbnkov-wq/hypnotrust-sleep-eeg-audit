from __future__ import annotations

from typing import Any

import mne
import pandas as pd

from core.config import get_default_pipeline_config
from core.feature_extraction import extract_temporal_features_from_raw
from core.prediction import run_prediction
from core.types import PipelineConfig, PipelineResult
from core.uncertainty_adapter import compute_uncertainty_dataframe


def _sanitize_labels_for_metrics(labels: list[str] | None) -> list[str] | None:
    """
    Return labels only if every epoch has a verified usable label.

    If unknown/movement labels exist, return None.
    This prevents accidental metric computation on placeholder labels.
    """

    if labels is None:
        return None

    if any(label is None for label in labels):  # type: ignore[comparison-overlap]
        return None

    return labels


def run_feature_dataframe_pipeline(
    feature_df: pd.DataFrame,
    labels: list[str] | None = None,
    config: PipelineConfig | None = None,
    recording_info: dict[str, Any] | None = None,
) -> PipelineResult:
    """
    Run the app-facing prediction pipeline from an existing feature DataFrame.

    This entry point is useful for:
    - demo mode with precomputed features
    - research mode with locked feature tables
    - tests

    It performs:
    - feature validation/reordering through prediction.py
    - probability prediction
    - predicted stage extraction
    - uncertainty scoring
    - PipelineResult construction

    It does not train, tune, or compute metrics.
    """

    if config is None:
        config = get_default_pipeline_config()

    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame.")

    if feature_df.empty:
        raise ValueError("feature_df is empty. Cannot run pipeline.")

    labels = _sanitize_labels_for_metrics(labels)

    probabilities, predicted_stages = run_prediction(
        feature_df=feature_df,
        config=config,
    )

    uncertainty_df = compute_uncertainty_dataframe(
        probabilities=probabilities,
        predicted_stages=predicted_stages,
        config=config,
    )

    if labels is not None and len(labels) != len(predicted_stages):
        raise ValueError(
            f"labels length ({len(labels)}) must match number of predictions "
            f"({len(predicted_stages)})."
        )

    info = dict(recording_info or {})
    info.update(
        {
            "n_epochs": len(predicted_stages),
            "epoch_length_sec": config.epoch_length_sec,
            "model_path": str(config.model_path),
            "uncertainty_policy": config.uncertainty_policy,
            "labels_available": labels is not None,
            "pipeline_entry": "run_feature_dataframe_pipeline",
        }
    )

    return PipelineResult(
        epochs_df=feature_df.reset_index(drop=True).copy(),
        predicted_stages=predicted_stages,
        probabilities=probabilities,
        uncertainty_df=uncertainty_df,
        labels=labels,
        stage_names=list(config.stage_order),
        recording_info=info,
        config_used=config,
    )


def run_full_pipeline(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations | None = None,
    config: PipelineConfig | None = None,
    *,
    subject: str | int = "uploaded",
    recording: str | int = 0,
) -> PipelineResult:
    """
    Run the full app-facing EDF pipeline.

    Flow:
    Raw PSG
    -> 13 Baseline v1 features
    -> 65 Temporal v2 features
    -> locked temporal v2 model
    -> predictions
    -> uncertainty table

    This function does not train, tune, overwrite artifacts, or compute metrics.
    """

    if config is None:
        config = get_default_pipeline_config()

    extraction = extract_temporal_features_from_raw(
        raw=raw,
        annotations=annotations,
        subject=subject,
        recording=recording,
        epoch_length_sec=config.epoch_length_sec,
    )

    labels = _sanitize_labels_for_metrics(extraction.labels)

    recording_info = {
        "pipeline_entry": "run_full_pipeline",
        "subject": subject,
        "recording": recording,
        "raw_channels": list(raw.ch_names),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "n_raw_samples": int(raw.n_times),
        "baseline_feature_rows": len(extraction.baseline_df),
        "temporal_feature_rows": len(extraction.temporal_df),
        "annotations_provided": annotations is not None,
        "labels_available": labels is not None,
    }

    return run_feature_dataframe_pipeline(
        feature_df=extraction.temporal_df,
        labels=labels,
        config=config,
        recording_info=recording_info,
    )