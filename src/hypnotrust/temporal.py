"""
Temporal-context feature construction for HypnoTrust.

The module creates feature vectors from neighbouring 30-second epochs
without using neighbouring sleep-stage labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypnotrust.constants import (
    BASELINE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
)
from hypnotrust.validation import validate_feature_dataframe


TEMPORAL_OFFSETS: tuple[int, ...] = (
    -2,
    -1,
    0,
    1,
    2,
)


def _format_offset(offset: int) -> str:
    """Return a stable textual representation of a temporal offset."""
    if offset > 0:
        return f"t+{offset}"

    return f"t{offset}"


TEMPORAL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{_format_offset(offset)}__{feature_name}"
    for offset in TEMPORAL_OFFSETS
    for feature_name in BASELINE_FEATURE_COLUMNS
)


def _validate_epoch_sequence(
    group: pd.DataFrame,
    *,
    subject: object,
    recording: object,
) -> None:
    """Require unique and consecutive epoch numbers within one recording."""
    epochs = pd.to_numeric(
        group["Epoch"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if len(np.unique(epochs)) != len(epochs):
        raise ValueError(
            "Duplicate epoch numbers found for "
            f"Subject={subject}, Recording={recording}."
        )

    if len(epochs) > 1:
        epoch_differences = np.diff(epochs)

        if not np.all(epoch_differences == 1):
            raise ValueError(
                "Epoch numbers must be consecutive within each recording. "
                f"Subject={subject}, Recording={recording}, "
                f"epochs={epochs.tolist()}."
            )


def build_temporal_feature_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create five-epoch temporal-context features.

    Each output row contains the 13 locked Baseline v1 features from:

    - ``t-2``
    - ``t-1``
    - ``t0``
    - ``t+1``
    - ``t+2``

    This creates 65 temporal features per epoch.

    Context is constructed separately for every Subject and Recording,
    so information never crosses participant or recording boundaries.

    At the beginning and end of a recording, the nearest available
    epoch is repeated. This preserves every original epoch.

    Only physiological feature values are shifted. The current epoch's
    metadata and sleep-stage label remain unchanged.

    Parameters
    ----------
    dataframe:
        Valid Baseline v1 feature table containing the four metadata
        columns and the 13 locked feature columns.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted table containing the four metadata
        columns followed by 65 temporal feature columns.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    ValueError
        If the Baseline v1 schema is invalid, or if epoch numbers within
        a recording are duplicated or non-consecutive.
    """
    validate_feature_dataframe(dataframe)

    sorted_dataframe = (
        dataframe
        .sort_values(
            by=[
                "Subject",
                "Recording",
                "Epoch",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    output_frames: list[pd.DataFrame] = []

    grouped = sorted_dataframe.groupby(
        [
            "Subject",
            "Recording",
        ],
        sort=False,
        dropna=False,
    )

    for (
        subject,
        recording,
    ), group in grouped:
        group = group.reset_index(drop=True)

        _validate_epoch_sequence(
            group,
            subject=subject,
            recording=recording,
        )

        base_features = group.loc[
            :,
            list(BASELINE_FEATURE_COLUMNS),
        ].to_numpy(
            dtype=float,
            copy=True,
        )

        number_of_epochs = len(group)
        base_indices = np.arange(number_of_epochs)

        temporal_data: dict[str, np.ndarray] = {}

        for offset in TEMPORAL_OFFSETS:
            shifted_indices = np.clip(
                base_indices + offset,
                0,
                number_of_epochs - 1,
            )

            shifted_features = base_features[
                shifted_indices
            ]

            for feature_index, feature_name in enumerate(
                BASELINE_FEATURE_COLUMNS
            ):
                temporal_column = (
                    f"{_format_offset(offset)}__{feature_name}"
                )

                temporal_data[temporal_column] = (
                    shifted_features[
                        :,
                        feature_index,
                    ]
                )

        metadata = group.loc[
            :,
            list(METADATA_COLUMNS),
        ].copy()

        temporal_features = pd.DataFrame(
            temporal_data
        )

        output_frames.append(
            pd.concat(
                [
                    metadata,
                    temporal_features,
                ],
                axis=1,
            )
        )

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    expected_columns = list(
        METADATA_COLUMNS
        + TEMPORAL_FEATURE_COLUMNS
    )

    return result.loc[
        :,
        expected_columns,
    ]