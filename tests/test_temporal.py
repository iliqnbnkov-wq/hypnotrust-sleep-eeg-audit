"""
Unit tests for HypnoTrust temporal-context feature construction.

The tests use only synthetic data and never modify real EEG data,
baseline artifacts, models, or result files.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
from hypnotrust.temporal import (
    TEMPORAL_FEATURE_COLUMNS,
    TEMPORAL_OFFSETS,
    build_temporal_feature_dataframe,
)


def feature_value(
    subject: int,
    recording_index: int,
    epoch: int,
    feature_index: int,
) -> float:
    """Create a unique deterministic value for one synthetic feature."""
    return float(
        subject * 100_000
        + recording_index * 10_000
        + epoch * 100
        + feature_index
    )


def make_group(
    *,
    subject: int = 0,
    recording_index: int = 0,
    epochs: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Create one valid synthetic Subject/Recording group."""
    rows: list[dict[str, object]] = []

    for row_index, epoch in enumerate(epochs):
        row: dict[str, object] = {
            "Subject": subject,
            "Recording": f"recording_{recording_index}",
            "Epoch": epoch,
            "Stage": STAGE_ORDER[
                row_index % len(STAGE_ORDER)
            ],
        }

        for feature_index, feature_name in enumerate(
            BASELINE_FEATURE_COLUMNS
        ):
            row[feature_name] = feature_value(
                subject,
                recording_index,
                epoch,
                feature_index,
            )

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=list(
            METADATA_COLUMNS
            + BASELINE_FEATURE_COLUMNS
        ),
    )


class TestTemporalFeatureContract(unittest.TestCase):
    """Tests for the locked Temporal v2 feature contract."""

    def test_temporal_feature_count_is_65(self) -> None:
        self.assertEqual(
            len(TEMPORAL_FEATURE_COLUMNS),
            65,
        )

    def test_temporal_offsets_are_locked(self) -> None:
        self.assertEqual(
            TEMPORAL_OFFSETS,
            (-2, -1, 0, 1, 2),
        )

    def test_output_columns_have_exact_order(self) -> None:
        dataframe = make_group()

        result = build_temporal_feature_dataframe(
            dataframe
        )

        expected_columns = list(
            METADATA_COLUMNS
            + TEMPORAL_FEATURE_COLUMNS
        )

        self.assertListEqual(
            list(result.columns),
            expected_columns,
        )

    def test_no_sleep_stage_label_leakage(self) -> None:
        dataframe = make_group()

        result = build_temporal_feature_dataframe(
            dataframe
        )

        temporal_columns = list(
            result.columns[
                len(METADATA_COLUMNS):
            ]
        )

        self.assertNotIn(
            "Stage",
            temporal_columns,
        )

        for column in temporal_columns:
            self.assertNotIn(
                "Stage",
                column,
            )


class TestTemporalConstruction(unittest.TestCase):
    """Tests for correct neighbouring-epoch feature construction."""

    def test_output_preserves_number_of_rows(self) -> None:
        dataframe = make_group()

        result = build_temporal_feature_dataframe(
            dataframe
        )

        self.assertEqual(
            len(result),
            len(dataframe),
        )

    def test_current_metadata_and_labels_are_preserved(self) -> None:
        dataframe = make_group()

        result = build_temporal_feature_dataframe(
            dataframe
        )

        expected_metadata = dataframe.loc[
            :,
            list(METADATA_COLUMNS),
        ].reset_index(drop=True)

        actual_metadata = result.loc[
            :,
            list(METADATA_COLUMNS),
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual_metadata,
            expected_metadata,
        )

    def test_middle_epoch_uses_correct_context(self) -> None:
        dataframe = make_group(
            epochs=(0, 1, 2, 3, 4)
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]
        middle_row = result.loc[
            result["Epoch"] == 2
        ].iloc[0]

        expected = {
            f"t-2__{feature_name}": feature_value(
                0, 0, 0, 0
            ),
            f"t-1__{feature_name}": feature_value(
                0, 0, 1, 0
            ),
            f"t0__{feature_name}": feature_value(
                0, 0, 2, 0
            ),
            f"t+1__{feature_name}": feature_value(
                0, 0, 3, 0
            ),
            f"t+2__{feature_name}": feature_value(
                0, 0, 4, 0
            ),
        }

        for column, expected_value in expected.items():
            self.assertEqual(
                middle_row[column],
                expected_value,
            )

    def test_start_boundary_repeats_first_epoch(self) -> None:
        dataframe = make_group(
            epochs=(0, 1, 2, 3, 4)
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]
        first_row = result.iloc[0]

        first_epoch_value = feature_value(
            0, 0, 0, 0
        )

        self.assertEqual(
            first_row[f"t-2__{feature_name}"],
            first_epoch_value,
        )
        self.assertEqual(
            first_row[f"t-1__{feature_name}"],
            first_epoch_value,
        )
        self.assertEqual(
            first_row[f"t0__{feature_name}"],
            first_epoch_value,
        )
        self.assertEqual(
            first_row[f"t+1__{feature_name}"],
            feature_value(0, 0, 1, 0),
        )
        self.assertEqual(
            first_row[f"t+2__{feature_name}"],
            feature_value(0, 0, 2, 0),
        )

    def test_end_boundary_repeats_last_epoch(self) -> None:
        dataframe = make_group(
            epochs=(0, 1, 2, 3, 4)
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]
        last_row = result.iloc[-1]

        last_epoch_value = feature_value(
            0, 0, 4, 0
        )

        self.assertEqual(
            last_row[f"t-2__{feature_name}"],
            feature_value(0, 0, 2, 0),
        )
        self.assertEqual(
            last_row[f"t-1__{feature_name}"],
            feature_value(0, 0, 3, 0),
        )
        self.assertEqual(
            last_row[f"t0__{feature_name}"],
            last_epoch_value,
        )
        self.assertEqual(
            last_row[f"t+1__{feature_name}"],
            last_epoch_value,
        )
        self.assertEqual(
            last_row[f"t+2__{feature_name}"],
            last_epoch_value,
        )

    def test_single_epoch_recording_repeats_same_epoch(self) -> None:
        dataframe = make_group(
            epochs=(0,)
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]
        expected_value = feature_value(
            0, 0, 0, 0
        )

        for prefix in (
            "t-2",
            "t-1",
            "t0",
            "t+1",
            "t+2",
        ):
            self.assertEqual(
                result.loc[
                    0,
                    f"{prefix}__{feature_name}",
                ],
                expected_value,
            )

    def test_unsorted_input_is_sorted_chronologically(self) -> None:
        dataframe = make_group(
            epochs=(0, 1, 2, 3, 4)
        )

        unsorted_dataframe = dataframe.iloc[
            [4, 1, 3, 0, 2]
        ].reset_index(drop=True)

        result = build_temporal_feature_dataframe(
            unsorted_dataframe
        )

        self.assertListEqual(
            list(result["Epoch"]),
            [0, 1, 2, 3, 4],
        )

    def test_input_dataframe_is_not_modified(self) -> None:
        dataframe = make_group()
        original = dataframe.copy(deep=True)

        build_temporal_feature_dataframe(
            dataframe
        )

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )


class TestTemporalBoundaries(unittest.TestCase):
    """Tests preventing context leakage across groups."""

    def test_context_does_not_cross_subject_boundary(self) -> None:
        subject_zero = make_group(
            subject=0,
            recording_index=0,
            epochs=(0, 1, 2),
        )

        subject_one = make_group(
            subject=1,
            recording_index=0,
            epochs=(0, 1, 2),
        )

        dataframe = pd.concat(
            [
                subject_zero,
                subject_one,
            ],
            ignore_index=True,
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]

        first_subject_one_row = result[
            result["Subject"] == 1
        ].iloc[0]

        expected_value = feature_value(
            1, 0, 0, 0
        )

        self.assertEqual(
            first_subject_one_row[
                f"t-2__{feature_name}"
            ],
            expected_value,
        )
        self.assertEqual(
            first_subject_one_row[
                f"t-1__{feature_name}"
            ],
            expected_value,
        )

    def test_context_does_not_cross_recording_boundary(self) -> None:
        recording_zero = make_group(
            subject=0,
            recording_index=0,
            epochs=(0, 1, 2),
        )

        recording_one = make_group(
            subject=0,
            recording_index=1,
            epochs=(0, 1, 2),
        )

        dataframe = pd.concat(
            [
                recording_zero,
                recording_one,
            ],
            ignore_index=True,
        )

        result = build_temporal_feature_dataframe(
            dataframe
        )

        feature_name = BASELINE_FEATURE_COLUMNS[0]

        first_recording_one_row = result[
            result["Recording"] == "recording_1"
        ].iloc[0]

        expected_value = feature_value(
            0, 1, 0, 0
        )

        self.assertEqual(
            first_recording_one_row[
                f"t-2__{feature_name}"
            ],
            expected_value,
        )
        self.assertEqual(
            first_recording_one_row[
                f"t-1__{feature_name}"
            ],
            expected_value,
        )


class TestTemporalValidation(unittest.TestCase):
    """Tests for invalid temporal input."""

    def test_missing_baseline_feature_raises(self) -> None:
        dataframe = make_group()

        dataframe = dataframe.drop(
            columns=[
                BASELINE_FEATURE_COLUMNS[0]
            ]
        )

        with self.assertRaises(ValueError):
            build_temporal_feature_dataframe(
                dataframe
            )

    def test_duplicate_epoch_raises(self) -> None:
        dataframe = make_group(
            epochs=(0, 1, 2)
        )

        dataframe.loc[
            1,
            "Epoch",
        ] = 0

        with self.assertRaises(ValueError):
            build_temporal_feature_dataframe(
                dataframe
            )

    def test_non_consecutive_epochs_raise(self) -> None:
        dataframe = make_group(
            epochs=(0, 2, 3)
        )

        with self.assertRaises(ValueError):
            build_temporal_feature_dataframe(
                dataframe
            )


if __name__ == "__main__":
    unittest.main()