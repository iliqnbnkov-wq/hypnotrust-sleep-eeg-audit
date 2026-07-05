from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import PipelineConfig, UncertaintyPolicy
from hypnotrust.constants import STAGE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "temporal_v2"
    / "sleep_stage_random_forest_temporal_v2.joblib"
)

DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "temporal_v2"
    / "sleep_stage_temporal_v2_features.json"
)

DEFAULT_METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "temporal_v2"
    / "sleep_stage_temporal_v2_metrics.json"
)

DEFAULT_CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "results"
    / "calibration_v3b"
    / "calibrator_v3b.joblib"
)

DEFAULT_EPOCH_LENGTH_SEC = 30
DEFAULT_REVIEW_FRACTION = 0.20
DEFAULT_UNCERTAINTY_POLICY: UncertaintyPolicy = "v3"


@dataclass(frozen=True)
class AppConfig:
    """
    App-level configuration for the HypnoTrust Streamlit prototype.

    This config uses the locked temporal v2 artifact by default.
    The app must not train or tune models.
    """

    project_root: Path
    model_path: Path
    features_path: Path
    metrics_path: Path
    epoch_length_sec: int = DEFAULT_EPOCH_LENGTH_SEC
    review_fraction: float = DEFAULT_REVIEW_FRACTION
    uncertainty_policy: UncertaintyPolicy = DEFAULT_UNCERTAINTY_POLICY
    apply_calibration: bool = False
    calibrator_path: Path | None = None


def _extract_feature_columns_from_json(data: Any) -> tuple[str, ...]:
    """
    Extract feature columns from a feature metadata JSON file.

    Supports common formats:
    - ["feature_a", "feature_b"]
    - {"feature_columns": [...]}
    - {"features": [...]}
    - {"columns": [...]}
    """

    if isinstance(data, list):
        return tuple(str(x) for x in data)

    if isinstance(data, dict):
        for key in ("feature_columns", "features", "columns", "selected_features"):
            value = data.get(key)
            if isinstance(value, list):
                return tuple(str(x) for x in value)

    raise ValueError(
        "Could not extract feature columns from feature JSON. "
        "Expected a list or a dict containing one of: "
        "feature_columns, features, columns, selected_features."
    )


def load_feature_columns(features_path: Path = DEFAULT_FEATURES_PATH) -> tuple[str, ...]:
    """
    Load ordered feature columns for the locked model artifact.

    The order matters. RandomForest inference must receive features in the
    exact same order used during training.
    """

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    data = json.loads(features_path.read_text(encoding="utf-8"))
    feature_columns = _extract_feature_columns_from_json(data)

    if not feature_columns:
        raise ValueError(f"No feature columns found in: {features_path}")

    return feature_columns


def get_default_app_config() -> AppConfig:
    """
    Return default local app configuration.

    MVP default:
    - temporal v2 model
    - v3 uncertainty
    - calibration disabled by default
    """

    return AppConfig(
        project_root=PROJECT_ROOT,
        model_path=DEFAULT_MODEL_PATH,
        features_path=DEFAULT_FEATURES_PATH,
        metrics_path=DEFAULT_METRICS_PATH,
        calibrator_path=DEFAULT_CALIBRATOR_PATH,
        apply_calibration=False,
    )


def get_default_pipeline_config() -> PipelineConfig:
    """
    Convert AppConfig into PipelineConfig used by core.pipeline.

    This function loads feature columns from the v2 features JSON.
    """

    app_config = get_default_app_config()
    feature_columns = load_feature_columns(app_config.features_path)

    return PipelineConfig(
        model_path=app_config.model_path,
        feature_columns=feature_columns,
        stage_order=tuple(STAGE_ORDER),
        calibrator_path=app_config.calibrator_path,
        apply_calibration=app_config.apply_calibration,
        uncertainty_policy=app_config.uncertainty_policy,
        epoch_length_sec=app_config.epoch_length_sec,
        channel_mapping=None,
    )


def validate_app_paths(config: AppConfig | None = None) -> list[str]:
    """
    Check whether required local files exist.

    Returns a list of problems.
    Empty list means the app config is usable.
    """

    if config is None:
        config = get_default_app_config()

    problems: list[str] = []

    if not config.model_path.exists():
        problems.append(f"Missing model artifact: {config.model_path}")

    if not config.features_path.exists():
        problems.append(f"Missing features file: {config.features_path}")

    if not config.metrics_path.exists():
        problems.append(f"Missing metrics file: {config.metrics_path}")

    if config.calibrator_path is not None and not config.calibrator_path.exists():
        problems.append(f"Missing calibrator artifact: {config.calibrator_path}")

    return problems