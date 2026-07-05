"""
Unit tests for HypnoTrust feature-table validation.

The tests use only synthetic data and never modify real baseline files.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from hypnotrust.constants import (
    BASELINE_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    STAGE_ORDER,
)
from hypnotrust.validation import validate_feature_dataframe


def make_valid_dataframe(
    subjects: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Create a small valid Baseline v1 feature table."""
    rows: list[dict[str, object]] = []

    for epoch, subject in enumerate(subjects):
        row: dict[str, object] = {
            "Subject": subject,
            "Recording": f"recording_{subject}",
            "Epoch": epoch,
            "Stage": STAGE_ORDER[
                epoch % len(STAGE_ORDER)
            ],
        }

        for feature_index, feature_name in enumerate(
            BASELINE_FEATURE_COLUMNS,
            start=1,
        ):
            row[feature_name] = (
                feature_index / 100.0
                + epoch / 1000.0
            )

        rows.append(row)

    expected_columns = list(
        METADATA_COLUMNS
        + BASELINE_FEATURE_COLUMNS
    )

    return pd.DataFrame(
        rows,
        columns=expected_columns,
    )


class TestValidFeatureDataFrame(unittest.TestCase):
    """Tests for valid feature tables."""

    def test_valid_dataframe_passes(self) -> None:
        dataframe = make_valid_dataframe()

        result = validate_feature_dataframe(
            dataframe
        )

        self.assertIsNone(result)

    def test_allowed_subjects_pass(self) -> None:
        dataframe = make_valid_dataframe(
            subjects=(0, 1, 2)
        )

        result = validate_feature_dataframe(
            dataframe,
            allowed_subjects={0, 1, 2},
        )

        self.assertIsNone(result)

    def test_all_sleep_stages_are_accepted(self) -> None:
        dataframe = make_valid_dataframe(
            subjects=(0, 1, 2, 3, 4)
        )

        dataframe["Stage"] = list(
            STAGE_ORDER
        )

        result = validate_feature_dataframe(
            dataframe
        )

        self.assertIsNone(result)


class TestInvalidInput(unittest.TestCase):
    """Tests for invalid function inputs."""

    def test_non_dataframe_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_feature_dataframe(
                {"Subject": [0]}
            )

    def test_empty_dataframe_raises_value_error(self) -> None:
        dataframe = (
            make_valid_dataframe()
            .iloc[0:0]
            .copy()
        )

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )


class TestColumnSchema(unittest.TestCase):
    """Tests for the locked Baseline v1 schema."""

    def test_missing_column_raises_value_error(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe = dataframe.drop(
            columns=[
                BASELINE_FEATURE_COLUMNS[0]
            ]
        )

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_extra_column_raises_value_error(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe["UnexpectedFeature"] = 1.0

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_wrong_column_order_raises_value_error(self) -> None:
        dataframe = make_valid_dataframe()

        columns = list(dataframe.columns)

        columns[0], columns[1] = (
            columns[1],
            columns[0],
        )

        dataframe = dataframe.loc[
            :,
            columns,
        ]

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_duplicate_columns_raise_value_error(self) -> None:
        dataframe = make_valid_dataframe()

        columns = list(dataframe.columns)
        columns[1] = columns[0]

        dataframe.columns = columns

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )


class TestMissingAndInvalidValues(unittest.TestCase):
    """Tests for invalid and non-finite values."""

    def test_missing_feature_value_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe.loc[
            0,
            BASELINE_FEATURE_COLUMNS[0],
        ] = np.nan

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_invalid_sleep_stage_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe.loc[
            0,
            "Stage",
        ] = "N4"

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_non_numeric_feature_raises(self) -> None:
        dataframe = make_valid_dataframe()

        feature_name = (
            BASELINE_FEATURE_COLUMNS[0]
        )

        dataframe[feature_name] = (
            dataframe[feature_name]
            .astype(object)
        )

        dataframe.loc[
            0,
            feature_name,
        ] = "invalid"

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_infinite_feature_value_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe.loc[
            0,
            BASELINE_FEATURE_COLUMNS[0],
        ] = np.inf

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )


class TestSubjectValidation(unittest.TestCase):
    """Tests for Subject identifiers."""

    def test_non_numeric_subject_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe["Subject"] = (
            dataframe["Subject"]
            .astype(object)
        )

        dataframe.loc[
            0,
            "Subject",
        ] = "subject_zero"

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_fractional_subject_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe["Subject"] = (
            dataframe["Subject"]
            .astype(object)
        )

        dataframe.loc[
            0,
            "Subject",
        ] = 1.5

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_negative_subject_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe.loc[
            0,
            "Subject",
        ] = -1

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_unexpected_subject_raises(self) -> None:
        dataframe = make_valid_dataframe(
            subjects=(0, 1, 9)
        )

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe,
                allowed_subjects={0, 1, 2},
            )


class TestEpochValidation(unittest.TestCase):
    """Tests for Epoch identifiers."""

    def test_non_numeric_epoch_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe["Epoch"] = (
            dataframe["Epoch"]
            .astype(object)
        )

        dataframe.loc[
            0,
            "Epoch",
        ] = "first"

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_fractional_epoch_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe["Epoch"] = (
            dataframe["Epoch"]
            .astype(object)
        )

        dataframe.loc[
            0,
            "Epoch",
        ] = 2.5

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )

    def test_negative_epoch_raises(self) -> None:
        dataframe = make_valid_dataframe()

        dataframe.loc[
            0,
            "Epoch",
        ] = -1

        with self.assertRaises(ValueError):
            validate_feature_dataframe(
                dataframe
            )


if __name__ == "__main__":
    unittest.main()