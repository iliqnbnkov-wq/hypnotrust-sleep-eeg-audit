"""
Stable data structures for the HypnoTrust app layer.

These types form the contract between core/ and app/.
No scientific logic lives here — only structure definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Mode = Literal["demo", "research", "user"]
UncertaintyPolicy = Literal["v3", "v4_stage_aware"]


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of EDF channel validation.
    """

    ok: bool
    missing_required_channels: list[str]
    available_channels: list[str]
    suggested_mapping: dict[str, str]
    message: str


@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration for one pipeline run.

    The app loads locked model artifacts only.
    It must not train or tune models.
    """

    model_path: Path
    feature_columns: tuple[str, ...]
    stage_order: tuple[str, ...]
    calibrator_path: Path | None = None
    apply_calibration: bool = False
    uncertainty_policy: UncertaintyPolicy = "v3"
    epoch_length_sec: int = 30
    channel_mapping: dict[str, str] | None = None


@dataclass
class PipelineResult:
    """
    Output of one complete app-facing pipeline run.

    If labels is None:
    - predictions can be shown
    - uncertainty can be shown
    - metrics must not be computed
    """

    epochs_df: pd.DataFrame
    predicted_stages: list[str]
    probabilities: np.ndarray
    uncertainty_df: pd.DataFrame
    labels: list[str] | None
    stage_names: list[str]
    recording_info: dict
    config_used: PipelineConfig


@dataclass
class ReviewQueueResult:
    """
    Output of review queue construction.
    """

    review_table: pd.DataFrame
    retained_table: pd.DataFrame
    confidence_threshold: float
    review_fraction: float
    n_review: int
    n_retained: int
    n_total: int