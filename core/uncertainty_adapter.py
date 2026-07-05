from __future__ import annotations

import numpy as np
import pandas as pd

from core.types import PipelineConfig


def compute_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute maximum class probability per epoch.

    This is model confidence, not clinical reliability.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    return np.max(probabilities, axis=1)


def compute_predictive_entropy(
    probabilities: np.ndarray,
    normalize: bool = True,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute predictive entropy per epoch.

    Higher entropy means the probability mass is more spread across classes.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    clipped = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)

    if normalize:
        max_entropy = np.log(probabilities.shape[1])
        entropy = entropy / max_entropy

    return entropy


def compute_margin(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute probability margin between the top two classes.

    Low margin means the model is choosing between close alternatives.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    if probabilities.shape[1] < 2:
        raise ValueError("At least two classes are required to compute margin.")

    sorted_probs = np.sort(probabilities, axis=1)

    return sorted_probs[:, -1] - sorted_probs[:, -2]


def compute_uncertainty_dataframe(
    probabilities: np.ndarray,
    predicted_stages: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Build per-epoch uncertainty table.

    Required output columns:
    - epoch
    - predicted_stage
    - confidence
    - entropy
    - margin
    - uncertainty_policy

    This table can be shown without ground-truth labels.
    It does not claim correctness.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    n_epochs = probabilities.shape[0]

    if len(predicted_stages) != n_epochs:
        raise ValueError(
            f"predicted_stages length ({len(predicted_stages)}) must match "
            f"probability rows ({n_epochs})."
        )

    if probabilities.shape[1] != len(config.stage_order):
        raise ValueError(
            f"probability columns ({probabilities.shape[1]}) must match "
            f"stage_order length ({len(config.stage_order)})."
        )

    confidence = compute_confidence(probabilities)
    entropy = compute_predictive_entropy(probabilities, normalize=True)
    margin = compute_margin(probabilities)

    return pd.DataFrame(
        {
            "epoch": list(range(n_epochs)),
            "predicted_stage": predicted_stages,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "uncertainty_policy": config.uncertainty_policy,
        }
    )