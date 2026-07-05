"""
Unit tests for hypnotrust.calibration.

All tests use synthetic data and do not read or modify project artifacts.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
N_STAGES = 5
STAGE_ORDER = ("Wake", "N1", "N2", "N3", "REM")


def _make_prob_matrix(
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a valid [n, 5] probability matrix summing to 1 per row."""
    random_generator = rng or RNG
    raw = random_generator.dirichlet(alpha=[2, 1, 3, 1, 2], size=n)
    return raw.astype(np.float64)


def _make_oof_df(
    n_per_subject: int = 40,
    subjects: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a minimal OOF DataFrame without Subject 9."""
    random_generator = rng or RNG
    rows = []

    for subject in subjects:
        probabilities = _make_prob_matrix(n_per_subject, random_generator)

        for epoch in range(n_per_subject):
            stage_index = int(np.argmax(probabilities[epoch]))

            rows.append(
                {
                    "Subject": subject,
                    "Recording": 1,
                    "Epoch": epoch,
                    "True_Stage": STAGE_ORDER[stage_index],
                    "Predicted_Stage_Uncalibrated": STAGE_ORDER[stage_index],
                    **{
                        f"Uncalibrated_Probability_{stage}": probabilities[
                            epoch,
                            index,
                        ]
                        for index, stage in enumerate(STAGE_ORDER)
                    },
                }
            )

    return pd.DataFrame(rows)


class TestReorderProba(unittest.TestCase):
    def test_reorder_reversed_classes(self) -> None:
        from hypnotrust.calibration import _reorder_proba_to_stage_order

        reversed_classes = np.array(["REM", "N3", "N2", "N1", "Wake"])
        probabilities = np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])

        result = _reorder_proba_to_stage_order(probabilities, reversed_classes)

        self.assertAlmostEqual(result[0, 0], 0.25)
        self.assertAlmostEqual(result[0, 1], 0.15)
        self.assertAlmostEqual(result[0, 2], 0.30)
        self.assertAlmostEqual(result[0, 3], 0.20)
        self.assertAlmostEqual(result[0, 4], 0.10)

    def test_reorder_missing_stage_raises(self) -> None:
        from hypnotrust.calibration import _reorder_proba_to_stage_order

        incomplete_classes = np.array(["Wake", "N1", "N2", "N3"])
        probabilities = np.ones((1, 4)) / 4

        with self.assertRaises(ValueError):
            _reorder_proba_to_stage_order(probabilities, incomplete_classes)

    def test_reorder_already_correct_order(self) -> None:
        from hypnotrust.calibration import _reorder_proba_to_stage_order

        classes = np.array(list(STAGE_ORDER))
        probabilities = _make_prob_matrix(10)

        result = _reorder_proba_to_stage_order(probabilities, classes)

        np.testing.assert_array_equal(result, probabilities)


class TestEncodeLabelToInt(unittest.TestCase):
    def test_correct_encoding(self) -> None:
        from hypnotrust.calibration import _encode_labels_to_int

        result = _encode_labels_to_int(["Wake", "N1", "N2", "N3", "REM"])

        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    def test_unknown_label_raises(self) -> None:
        from hypnotrust.calibration import _encode_labels_to_int

        with self.assertRaises(ValueError):
            _encode_labels_to_int(["Wake", "NREM", "N2"])

    def test_repeated_labels(self) -> None:
        from hypnotrust.calibration import _encode_labels_to_int

        result = _encode_labels_to_int(["REM", "REM", "Wake", "N3"])

        np.testing.assert_array_equal(result, [4, 4, 0, 3])


class TestFitCalibrator(unittest.TestCase):
    def setUp(self) -> None:
        self.oof_df = _make_oof_df()
        self.probability_columns = [
            f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER
        ]

    def test_returns_logistic_regression(self) -> None:
        from hypnotrust.calibration import fit_calibrator
        from sklearn.linear_model import LogisticRegression

        calibrator = fit_calibrator(
            self.oof_df,
            self.probability_columns,
            "True_Stage",
        )

        self.assertIsInstance(calibrator, LogisticRegression)

    def test_calibrator_classes_are_integers_in_stage_order(self) -> None:
        from hypnotrust.calibration import fit_calibrator

        calibrator = fit_calibrator(
            self.oof_df,
            self.probability_columns,
            "True_Stage",
        )

        np.testing.assert_array_equal(calibrator.classes_, [0, 1, 2, 3, 4])

    def test_wrong_number_of_prob_columns_raises(self) -> None:
        from hypnotrust.calibration import fit_calibrator

        with self.assertRaises(ValueError):
            fit_calibrator(
                self.oof_df,
                self.probability_columns[:3],
                "True_Stage",
            )

    def test_subject_9_raises(self) -> None:
        from hypnotrust.calibration import fit_calibrator

        df_with_9 = _make_oof_df(subjects=(0, 1, 9))

        with self.assertRaises(ValueError):
            fit_calibrator(
                df_with_9,
                self.probability_columns,
                "True_Stage",
            )

    def test_unknown_label_raises(self) -> None:
        from hypnotrust.calibration import fit_calibrator

        bad_dataframe = self.oof_df.copy()
        bad_dataframe.loc[0, "True_Stage"] = "NREM"

        with self.assertRaises(ValueError):
            fit_calibrator(
                bad_dataframe,
                self.probability_columns,
                "True_Stage",
            )


class TestApplyCalibrator(unittest.TestCase):
    def setUp(self) -> None:
        from hypnotrust.calibration import fit_calibrator

        self.oof_df = _make_oof_df()
        self.probability_columns = [
            f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER
        ]
        self.calibrator = fit_calibrator(
            self.oof_df,
            self.probability_columns,
            "True_Stage",
        )

    def test_output_shape_preserved(self) -> None:
        from hypnotrust.calibration import apply_calibrator

        probabilities = _make_prob_matrix(50)
        result = apply_calibrator(self.calibrator, probabilities)

        self.assertEqual(result.shape, (50, 5))

    def test_output_rows_sum_to_one(self) -> None:
        from hypnotrust.calibration import apply_calibrator

        probabilities = _make_prob_matrix(100)
        result = apply_calibrator(self.calibrator, probabilities)

        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_wrong_shape_raises(self) -> None:
        from hypnotrust.calibration import apply_calibrator

        with self.assertRaises(ValueError):
            apply_calibrator(self.calibrator, np.ones((10, 3)) / 3)

    def test_1d_input_raises(self) -> None:
        from hypnotrust.calibration import apply_calibrator

        with self.assertRaises(ValueError):
            apply_calibrator(self.calibrator, np.ones(5) / 5)

    def test_output_values_in_zero_one(self) -> None:
        from hypnotrust.calibration import apply_calibrator

        probabilities = _make_prob_matrix(80)
        result = apply_calibrator(self.calibrator, probabilities)

        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_wrong_calibrator_classes_raises(self) -> None:
        from hypnotrust.calibration import apply_calibrator
        from sklearn.linear_model import LogisticRegression

        bad_calibrator = LogisticRegression()
        bad_calibrator.classes_ = np.array([0, 1, 2])

        with self.assertRaises(ValueError):
            apply_calibrator(bad_calibrator, _make_prob_matrix(10))

    def test_calibrator_without_classes_attr_raises(self) -> None:
        from hypnotrust.calibration import apply_calibrator
        from sklearn.linear_model import LogisticRegression

        unfitted = LogisticRegression()

        with self.assertRaises(ValueError):
            apply_calibrator(unfitted, _make_prob_matrix(10))


class TestGenerateOofProbabilities(unittest.TestCase):
    def _make_df(
        self,
        subjects: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        stages = list(STAGE_ORDER)

        for subject in subjects:
            for epoch in range(25):
                rows.append(
                    {
                        "Subject": subject,
                        "Recording": 1,
                        "Epoch": epoch,
                        "Stage": stages[epoch % 5],
                        "feat_a": rng.normal(),
                        "feat_b": rng.normal(),
                    }
                )

        return pd.DataFrame(rows)

    def _configs(self) -> tuple[dict, dict]:
        model_config = {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        }
        cv_config = {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        }

        return model_config, cv_config

    def test_output_columns_present(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df()
        model_config, cv_config = self._configs()

        result = generate_oof_probabilities(
            dataframe,
            ["feat_a", "feat_b"],
            "Stage",
            "Subject",
            model_config,
            cv_config,
        )

        expected_columns = [
            "Subject",
            "Recording",
            "Epoch",
            "True_Stage",
            "Predicted_Stage_Uncalibrated",
            *[f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER],
        ]

        for column in expected_columns:
            self.assertIn(column, result.columns, msg=f"Missing column: {column}")

    def test_row_count_matches_input(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df()
        model_config, cv_config = self._configs()

        result = generate_oof_probabilities(
            dataframe,
            ["feat_a", "feat_b"],
            "Stage",
            "Subject",
            model_config,
            cv_config,
        )

        self.assertEqual(len(result), len(dataframe))

    def test_probabilities_sum_to_one(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df()
        model_config, cv_config = self._configs()

        result = generate_oof_probabilities(
            dataframe,
            ["feat_a", "feat_b"],
            "Stage",
            "Subject",
            model_config,
            cv_config,
        )

        probability_columns = [
            f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER
        ]

        np.testing.assert_allclose(
            result[probability_columns].sum(axis=1).to_numpy(),
            1.0,
            atol=1e-5,
        )

    def test_no_nan_in_output(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df()
        model_config, cv_config = self._configs()

        result = generate_oof_probabilities(
            dataframe,
            ["feat_a", "feat_b"],
            "Stage",
            "Subject",
            model_config,
            cv_config,
        )

        probability_columns = [
            f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER
        ]

        self.assertFalse(result[probability_columns].isna().any().any())

    def test_subject_9_raises(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df(subjects=(0, 1, 9))
        model_config, cv_config = self._configs()

        with self.assertRaises(ValueError):
            generate_oof_probabilities(
                dataframe,
                ["feat_a", "feat_b"],
                "Stage",
                "Subject",
                model_config,
                cv_config,
            )

    def test_probability_columns_in_stage_order(self) -> None:
        from hypnotrust.calibration import generate_oof_probabilities

        dataframe = self._make_df()
        model_config, cv_config = self._configs()

        result = generate_oof_probabilities(
            dataframe,
            ["feat_a", "feat_b"],
            "Stage",
            "Subject",
            model_config,
            cv_config,
        )

        probability_columns = [
            f"Uncalibrated_Probability_{stage}" for stage in STAGE_ORDER
        ]

        column_positions = [list(result.columns).index(col) for col in probability_columns]

        self.assertEqual(column_positions, sorted(column_positions))


class TestComputeCalibrationComparison(unittest.TestCase):
    def setUp(self) -> None:
        n = 200
        rng = np.random.default_rng(7)

        self.uncalibrated_probs = _make_prob_matrix(n, rng)
        self.calibrated_probs = _make_prob_matrix(n, rng)
        self.y_true = rng.integers(0, N_STAGES, size=n)
        self.label_names = list(STAGE_ORDER)

    def test_output_keys_present(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        self.assertIn("uncalibrated", result)
        self.assertIn("calibrated", result)
        self.assertIn("n_labels_changed", result)
        self.assertIn("selective_prediction", result)

    def test_metric_keys_present(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        required = {
            "ece",
            "log_loss",
            "brier_score",
            "mean_confidence_correct",
            "mean_confidence_incorrect",
            "mean_entropy_correct",
            "mean_entropy_incorrect",
            "accuracy",
            "macro_f1",
            "kappa",
        }

        for split in ("uncalibrated", "calibrated"):
            missing = required - set(result[split].keys())
            self.assertFalse(missing, msg=f"{split} missing keys: {missing}")

    def test_n_labels_changed_is_nonnegative_int(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        self.assertIsInstance(result["n_labels_changed"], int)
        self.assertGreaterEqual(result["n_labels_changed"], 0)

    def test_n_labels_changed_zero_when_identical(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.uncalibrated_probs,
            self.label_names,
        )

        self.assertEqual(result["n_labels_changed"], 0)

    def test_selective_prediction_both_keys(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        selective_prediction = result["selective_prediction"]

        self.assertIn("uncalibrated", selective_prediction)
        self.assertIn("calibrated", selective_prediction)
        self.assertIsInstance(selective_prediction["uncalibrated"], list)
        self.assertIsInstance(selective_prediction["calibrated"], list)

    def test_metrics_are_finite(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        for split in ("uncalibrated", "calibrated"):
            for key, value in result[split].items():
                if isinstance(value, float):
                    self.assertTrue(
                        np.isfinite(value),
                        msg=f"{split}.{key} is not finite: {value}",
                    )

    def test_accuracy_in_valid_range(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        for split in ("uncalibrated", "calibrated"):
            accuracy = result[split]["accuracy"]
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_ece_nonnegative(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        result = compute_calibration_comparison(
            self.y_true,
            self.uncalibrated_probs,
            self.calibrated_probs,
            self.label_names,
        )

        for split in ("uncalibrated", "calibrated"):
            self.assertGreaterEqual(result[split]["ece"], 0.0)

    def test_y_true_out_of_range_raises(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        bad_y = self.y_true.copy()
        bad_y[0] = 10

        with self.assertRaises(ValueError):
            compute_calibration_comparison(
                bad_y,
                self.uncalibrated_probs,
                self.calibrated_probs,
                self.label_names,
            )

    def test_y_true_negative_raises(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        bad_y = self.y_true.copy()
        bad_y[5] = -1

        with self.assertRaises(ValueError):
            compute_calibration_comparison(
                bad_y,
                self.uncalibrated_probs,
                self.calibrated_probs,
                self.label_names,
            )

    def test_wrong_label_names_length_raises(self) -> None:
        from hypnotrust.calibration import compute_calibration_comparison

        with self.assertRaises(ValueError):
            compute_calibration_comparison(
                self.y_true,
                self.uncalibrated_probs,
                self.calibrated_probs,
                label_names=["Wake", "N1", "N2"],
            )


if __name__ == "__main__":
    unittest.main()
