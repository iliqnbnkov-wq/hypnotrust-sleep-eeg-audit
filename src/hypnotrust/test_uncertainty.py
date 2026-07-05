"""
Unit tests for HypnoTrust uncertainty and calibration utilities.

The tests use synthetic probability distributions and never modify
real EEG data, trained models, or result artifacts.
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


from hypnotrust.uncertainty import (
    build_uncertainty_dataframe,
    calibration_table,
    expected_calibration_error,
    maximum_confidence,
    multiclass_brier_score,
    multiclass_log_loss,
    predicted_class_indices,
    predictive_entropy,
    selective_performance_curve,
    validate_probability_matrix,
)


class TestProbabilityMatrixValidation(unittest.TestCase):
    """Tests for probability-matrix validation."""

    def test_valid_matrix_is_returned_as_float_array(self) -> None:
        probabilities = [
            [0.8, 0.2],
            [0.3, 0.7],
        ]

        result = validate_probability_matrix(
            probabilities
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, float)
        self.assertEqual(result.shape, (2, 2))

        np.testing.assert_allclose(
            result,
            probabilities,
        )

    def test_one_dimensional_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [0.8, 0.2]
            )

    def test_empty_matrix_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                np.empty((0, 2))
            )

    def test_single_class_matrix_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [[1.0], [1.0]]
            )

    def test_nan_probability_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [
                    [np.nan, 0.0],
                    [0.4, 0.6],
                ]
            )

    def test_infinite_probability_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [
                    [np.inf, 0.0],
                    [0.4, 0.6],
                ]
            )

    def test_probability_outside_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [
                    [1.1, -0.1],
                    [0.4, 0.6],
                ]
            )

    def test_rows_not_summing_to_one_raise(self) -> None:
        with self.assertRaises(ValueError):
            validate_probability_matrix(
                [
                    [0.6, 0.5],
                    [0.4, 0.6],
                ]
            )


class TestBasicUncertaintyMetrics(unittest.TestCase):
    """Tests for confidence, class prediction, and entropy."""

    def test_maximum_confidence(self) -> None:
        probabilities = [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.5, 0.5],
        ]

        result = maximum_confidence(
            probabilities
        )

        np.testing.assert_allclose(
            result,
            [0.8, 0.6, 0.5],
        )

    def test_predicted_class_indices(self) -> None:
        probabilities = [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.1, 0.9],
        ]

        result = predicted_class_indices(
            probabilities
        )

        np.testing.assert_array_equal(
            result,
            [0, 1, 1],
        )

    def test_deterministic_prediction_has_zero_entropy(self) -> None:
        probabilities = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        result = predictive_entropy(
            probabilities,
            normalize=True,
        )

        np.testing.assert_allclose(
            result,
            [0.0, 0.0],
            atol=1e-12,
        )

    def test_uniform_prediction_has_normalized_entropy_one(self) -> None:
        probabilities = [
            [1 / 3, 1 / 3, 1 / 3],
        ]

        result = predictive_entropy(
            probabilities,
            normalize=True,
        )

        np.testing.assert_allclose(
            result,
            [1.0],
            atol=1e-12,
        )

    def test_uniform_non_normalized_entropy_equals_log_classes(self) -> None:
        probabilities = [
            [0.25, 0.25, 0.25, 0.25],
        ]

        result = predictive_entropy(
            probabilities,
            normalize=False,
        )

        np.testing.assert_allclose(
            result,
            [np.log(4.0)],
            atol=1e-12,
        )

    def test_build_uncertainty_dataframe(self) -> None:
        probabilities = [
            [0.8, 0.2],
            [0.4, 0.6],
        ]

        classes = [
            "Wake",
            "REM",
        ]

        result = build_uncertainty_dataframe(
            probabilities,
            classes,
        )

        self.assertListEqual(
            list(result.columns),
            [
                "Predicted_Class",
                "Maximum_Confidence",
                "Predictive_Entropy",
            ],
        )

        self.assertListEqual(
            list(result["Predicted_Class"]),
            ["Wake", "REM"],
        )

        np.testing.assert_allclose(
            result["Maximum_Confidence"],
            [0.8, 0.6],
        )

        self.assertTrue(
            result["Predictive_Entropy"]
            .between(0.0, 1.0)
            .all()
        )


class TestClassValidation(unittest.TestCase):
    """Tests for class-label validation."""

    def test_class_count_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_uncertainty_dataframe(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.7, 0.1],
                ],
                ["Wake", "REM"],
            )

    def test_duplicate_class_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_uncertainty_dataframe(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                ],
                ["Wake", "Wake"],
            )

    def test_missing_class_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_uncertainty_dataframe(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                ],
                ["Wake", np.nan],
            )


class TestProbabilisticScores(unittest.TestCase):
    """Tests for multiclass log loss and Brier score."""

    def test_multiclass_log_loss_known_value(self) -> None:
        y_true = [
            "A",
            "B",
        ]

        probabilities = [
            [0.8, 0.2],
            [0.3, 0.7],
        ]

        expected = -(
            np.log(0.8)
            + np.log(0.7)
        ) / 2

        result = multiclass_log_loss(
            y_true,
            probabilities,
            classes=["A", "B"],
        )

        self.assertAlmostEqual(
            result,
            expected,
            places=12,
        )

    def test_multiclass_brier_score_known_value(self) -> None:
        y_true = [
            "A",
            "B",
        ]

        probabilities = [
            [0.8, 0.2],
            [0.3, 0.7],
        ]

        result = multiclass_brier_score(
            y_true,
            probabilities,
            classes=["A", "B"],
        )

        self.assertAlmostEqual(
            result,
            0.13,
            places=12,
        )

    def test_log_loss_unknown_true_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_log_loss(
                ["A", "Unknown"],
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
                classes=["A", "B"],
            )

    def test_brier_score_unknown_true_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_brier_score(
                ["A", "Unknown"],
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
                classes=["A", "B"],
            )

    def test_invalid_log_loss_epsilon_raises(self) -> None:
        for epsilon in (
            0.0,
            1.0,
            -0.1,
            1.1,
        ):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    multiclass_log_loss(
                        ["A"],
                        [[0.8, 0.2]],
                        classes=["A", "B"],
                        epsilon=epsilon,
                    )

    def test_true_label_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            multiclass_brier_score(
                ["A"],
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ],
                classes=["A", "B"],
            )


class TestCalibrationMetrics(unittest.TestCase):
    """Tests for calibration tables and ECE."""

    def test_calibration_table_known_values(self) -> None:
        y_true = [
            "A",
            "B",
            "A",
            "B",
        ]

        y_pred = [
            "A",
            "A",
            "A",
            "A",
        ]

        confidence = [
            0.2,
            0.4,
            0.6,
            0.8,
        ]

        result = calibration_table(
            y_true,
            y_pred,
            confidence,
            number_of_bins=2,
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertListEqual(
            list(result["Count"]),
            [2, 2],
        )

        np.testing.assert_allclose(
            result["Mean_Confidence"],
            [0.3, 0.7],
        )

        np.testing.assert_allclose(
            result["Accuracy"],
            [0.5, 0.5],
        )

        np.testing.assert_allclose(
            result["Calibration_Gap"],
            [0.2, 0.2],
        )

    def test_expected_calibration_error_known_value(self) -> None:
        result = expected_calibration_error(
            y_true=["A", "B", "A", "B"],
            y_pred=["A", "A", "A", "A"],
            confidence=[0.2, 0.4, 0.6, 0.8],
            number_of_bins=2,
        )

        self.assertAlmostEqual(
            result,
            0.2,
            places=12,
        )

    def test_two_dimensional_confidence_raises(self) -> None:
        with self.assertRaises(ValueError):
            calibration_table(
                y_true=["A", "B"],
                y_pred=["A", "B"],
                confidence=[[0.8], [0.7]],
            )

    def test_confidence_outside_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            calibration_table(
                y_true=["A", "B"],
                y_pred=["A", "B"],
                confidence=[1.1, 0.7],
            )

    def test_invalid_number_of_bins_raises(self) -> None:
        for number_of_bins in (
            0,
            -1,
            True,
        ):
            with self.subTest(
                number_of_bins=number_of_bins
            ):
                with self.assertRaises(ValueError):
                    calibration_table(
                        y_true=["A", "B"],
                        y_pred=["A", "B"],
                        confidence=[0.8, 0.7],
                        number_of_bins=number_of_bins,
                    )

    def test_calibration_label_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            calibration_table(
                y_true=["A"],
                y_pred=["A", "B"],
                confidence=[0.8, 0.7],
            )


class TestSelectivePrediction(unittest.TestCase):
    """Tests for confidence-based selective prediction."""

    def test_selective_curve_known_values(self) -> None:
        result = selective_performance_curve(
            y_true=["A", "A", "B", "B"],
            y_pred=["A", "A", "A", "A"],
            confidence=[0.9, 0.8, 0.7, 0.6],
            coverages=(1.0, 0.75, 0.5),
        )

        self.assertListEqual(
            list(result["Retained_Count"]),
            [4, 3, 2],
        )

        self.assertListEqual(
            list(result["Rejected_Count"]),
            [0, 1, 2],
        )

        np.testing.assert_allclose(
            result["Actual_Coverage"],
            [1.0, 0.75, 0.5],
        )

        np.testing.assert_allclose(
            result["Accuracy"],
            [
                0.5,
                2 / 3,
                1.0,
            ],
        )

        np.testing.assert_allclose(
            result["Risk"],
            [
                0.5,
                1 / 3,
                0.0,
            ],
        )

        np.testing.assert_allclose(
            result["Confidence_Threshold"],
            [
                0.6,
                0.7,
                0.8,
            ],
        )

    def test_empty_coverages_raise(self) -> None:
        with self.assertRaises(ValueError):
            selective_performance_curve(
                y_true=["A", "B"],
                y_pred=["A", "B"],
                confidence=[0.8, 0.7],
                coverages=(),
            )

    def test_invalid_coverage_values_raise(self) -> None:
        for coverages in (
            (0.0,),
            (-0.1,),
            (1.1,),
        ):
            with self.subTest(
                coverages=coverages
            ):
                with self.assertRaises(ValueError):
                    selective_performance_curve(
                        y_true=["A", "B"],
                        y_pred=["A", "B"],
                        confidence=[0.8, 0.7],
                        coverages=coverages,
                    )

    def test_invalid_selective_confidence_raises(self) -> None:
        with self.assertRaises(ValueError):
            selective_performance_curve(
                y_true=["A", "B"],
                y_pred=["A", "B"],
                confidence=[0.8, np.nan],
            )

    def test_selective_label_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            selective_performance_curve(
                y_true=["A"],
                y_pred=["A", "B"],
                confidence=[0.8, 0.7],
            )


if __name__ == "__main__":
    unittest.main()