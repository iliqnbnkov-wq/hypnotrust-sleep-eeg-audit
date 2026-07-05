"""
Validation utilities for HypnoTrust feature tables.

The module verifies that sleep-staging feature tables follow the locked
Baseline v1 schema before they are used for training or evaluation.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from hypnotrust.constants import (
    BASELINE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    STAGE_ORDER,
)


EXPECTED_BASELINE_COLUMNS = (
    METADATA_COLUMNS + BASELINE_FEATURE_COLUMNS
)


def _require_integer_like(
    series: pd.Series,
    column_name: str,
) -> None:
    """Require all values in a Series to represent integers."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains "
            "non-numeric or missing values."
        )

    if not np.equal(
        numeric.to_numpy(),
        np.floor(numeric.to_numpy()),
    ).all():
        raise ValueError(
            f"Column '{column_name}' must contain "
            "integer-like values."
        )


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
    *,
    allowed_subjects: Collection[int] | None = None,
) -> None:
    """Validate a HypnoTrust Baseline v1 feature table.

    Parameters
    ----------
    dataframe:
        DataFrame containing metadata and the 13 locked baseline
        characteristics.

    allowed_subjects:
        Optional collection of permitted subject IDs. When omitted,
        the subject IDs are not restricted.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    ValueError
        If the table is empty, contains incorrect columns, invalid
        sleep stages, missing values, non-finite feature values, or
        invalid subject and epoch identifiers.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Feature DataFrame must not be empty."
        )

    if dataframe.columns.duplicated().any():
        duplicated = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate columns found: {duplicated}"
        )

    expected_columns = list(
        EXPECTED_BASELINE_COLUMNS
    )

    actual_columns = list(
        dataframe.columns
    )

    if actual_columns != expected_columns:
        missing_columns = [
            column
            for column in expected_columns
            if column not in actual_columns
        ]

        extra_columns = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        raise ValueError(
            "Baseline feature schema mismatch. "
            f"Missing columns: {missing_columns}. "
            f"Extra columns: {extra_columns}. "
            "Column order must match the locked "
            "Baseline v1 schema."
        )

    if dataframe.isna().any().any():
        columns_with_missing_values = (
            dataframe.columns[
                dataframe.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Missing values found in columns: "
            f"{columns_with_missing_values}"
        )

    invalid_stages = sorted(
        set(dataframe["Stage"])
        - set(STAGE_ORDER)
    )

    if invalid_stages:
        raise ValueError(
            "Invalid sleep stages found: "
            f"{invalid_stages}"
        )

    _require_integer_like(
        dataframe["Subject"],
        "Subject",
    )

    _require_integer_like(
        dataframe["Epoch"],
        "Epoch",
    )

    if (
        pd.to_numeric(
            dataframe["Subject"]
        )
        < 0
    ).any():
        raise ValueError(
            "Subject IDs must not be negative."
        )

    if (
        pd.to_numeric(
            dataframe["Epoch"]
        )
        < 0
    ).any():
        raise ValueError(
            "Epoch values must not be negative."
        )

    for feature_column in BASELINE_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(
            dataframe[feature_column]
        ):
            raise ValueError(
                f"Feature column '{feature_column}' "
                "must be numeric."
            )

    feature_values = dataframe[
        list(BASELINE_FEATURE_COLUMNS)
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Feature table contains infinite "
            "or non-finite values."
        )

    if allowed_subjects is not None:
        allowed_subject_set = {
            int(subject)
            for subject in allowed_subjects
        }

        present_subjects = {
            int(subject)
            for subject in dataframe["Subject"]
        }

        unexpected_subjects = sorted(
            present_subjects
            - allowed_subject_set
        )

        if unexpected_subjects:
            raise ValueError(
                "Unexpected subject IDs found: "
                f"{unexpected_subjects}"
            )