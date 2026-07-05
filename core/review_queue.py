from __future__ import annotations

import math

import pandas as pd

from core.types import ReviewQueueResult


_REQUIRED_UNCERTAINTY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "confidence",
    "entropy",
)


def build_review_queue(
    uncertainty_df: pd.DataFrame,
    predicted_stages: list[str],
    fraction: float,
    labels: list[str] | None = None,
) -> ReviewQueueResult:
    """
    Build a human-review queue by ranking epochs on ascending confidence.

    The review queue contains the lowest-confidence epochs.
    This function does not compute accuracy, F1, kappa, or clinical metrics.
    """

    for col in _REQUIRED_UNCERTAINTY_COLUMNS:
        if col not in uncertainty_df.columns:
            raise ValueError(
                f"uncertainty_df is missing required column '{col}'. "
                f"Available columns: {list(uncertainty_df.columns)}"
            )

    if len(predicted_stages) != len(uncertainty_df):
        raise ValueError(
            f"predicted_stages length ({len(predicted_stages)}) must match "
            f"uncertainty_df rows ({len(uncertainty_df)})."
        )

    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}.")

    if labels is not None and len(labels) != len(uncertainty_df):
        raise ValueError(
            f"labels length ({len(labels)}) must match "
            f"uncertainty_df rows ({len(uncertainty_df)})."
        )

    n_total = len(uncertainty_df)
    n_review = math.ceil(fraction * n_total)
    n_review = min(n_review, n_total)

    df = uncertainty_df.copy().reset_index(drop=True)
    df["predicted_stage"] = predicted_stages

    if labels is not None:
        df["true_stage"] = labels

    df_sorted = df.sort_values("confidence", ascending=True).reset_index(drop=True)

    review_df = df_sorted.iloc[:n_review].copy()
    retained_df = df_sorted.iloc[n_review:].copy()

    review_df["review_flag"] = True
    retained_df["review_flag"] = False

    confidence_threshold = (
        float(review_df["confidence"].max()) if n_review > 0 else 0.0
    )

    return ReviewQueueResult(
        review_table=review_df.reset_index(drop=True),
        retained_table=retained_df.reset_index(drop=True),
        confidence_threshold=confidence_threshold,
        review_fraction=fraction,
        n_review=n_review,
        n_retained=n_total - n_review,
        n_total=n_total,
    )