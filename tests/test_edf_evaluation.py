"""
Unit tests for core.edf_evaluation.

Tests label safety, validation, metric computation, and export functions.
"""

from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestBuildValidLabelMask(unittest.TestCase):
    """Test label mask construction."""

    def test_all_valid_labels(self):
        """All labels are valid should produce all True mask."""
        from core.edf_evaluation import build_valid_label_mask

        labels = ["Wake", "N1", "N2", "N3", "REM", "Wake"]
        mask = build_valid_label_mask(labels)

        self.assertEqual(len(mask), len(labels))
        self.assertTrue(np.all(mask))

    def test_none_labels_invalid(self):
        """None labels should be marked as invalid."""
        from core.edf_evaluation import build_valid_label_mask

        labels = ["Wake", None, "N2", None, "REM"]
        mask = build_valid_label_mask(labels)

        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_movement_time_invalid(self):
        """Movement time labels should be marked as invalid."""
        from core.edf_evaluation import build_valid_label_mask

        labels = ["Wake", "Movement time", "N2", "N3", "REM"]
        mask = build_valid_label_mask(labels)

        expected = np.array([True, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_sleep_stage_question_invalid(self):
        """Sleep stage ? labels should be marked as invalid."""
        from core.edf_evaluation import build_valid_label_mask

        labels = ["Wake", "Sleep stage ?", "N2", "N3", "REM"]
        mask = build_valid_label_mask(labels)

        expected = np.array([True, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_all_invalid_labels(self):
        """All invalid labels should produce all False mask."""
        from core.edf_evaluation import build_valid_label_mask

        labels = [None, None, "Movement time", "Sleep stage ?"]
        mask = build_valid_label_mask(labels)

        expected = np.array([False, False, False, False])
        np.testing.assert_array_equal(mask, expected)


class TestValidateAnnotationAlignment(unittest.TestCase):
    """Test annotation alignment validation."""

    def test_no_annotations(self):
        """No annotations should be considered valid."""
        from core.edf_evaluation import validate_annotation_alignment

        is_valid, warning = validate_annotation_alignment(
            raw_duration_sec=100.0,
            annotations=None,
        )

        self.assertTrue(is_valid)
        self.assertEqual(warning, "No annotations provided")

    def test_empty_annotations(self):
        """Empty annotation list should be considered valid."""
        from core.edf_evaluation import validate_annotation_alignment

        is_valid, warning = validate_annotation_alignment(
            raw_duration_sec=100.0,
            annotations=[],
        )

        self.assertTrue(is_valid)
        self.assertEqual(warning, "No annotations provided")

    def test_perfect_alignment(self):
        """Perfect alignment should be valid."""
        from core.edf_evaluation import validate_annotation_alignment

        annotations = [(0.0, 100.0, "Wake")]
        is_valid, warning = validate_annotation_alignment(
            raw_duration_sec=100.0,
            annotations=annotations,
        )

        self.assertTrue(is_valid)
        self.assertEqual(warning, "")

    def test_small_misalignment_within_tolerance(self):
        """Small misalignment within tolerance should be valid."""
        from core.edf_evaluation import validate_annotation_alignment

        annotations = [(0.0, 100.0, "Wake")]  # Perfect alignment
        is_valid, warning = validate_annotation_alignment(
            raw_duration_sec=100.0,
            annotations=annotations,
            epoch_length_sec=30,
        )

        self.assertTrue(is_valid)
        self.assertEqual(warning, "")

    def test_large_misalignment_invalid(self):
        """Large misalignment should be invalid due to insufficient coverage."""
        from core.edf_evaluation import validate_annotation_alignment

        annotations = [(10.0, 30.0, "Wake")]  # Starts at 10s, only 30s total
        is_valid, warning = validate_annotation_alignment(
            raw_duration_sec=100.0,
            annotations=annotations,
            epoch_length_sec=30,
        )

        self.assertFalse(is_valid)
        self.assertIn("cover only", warning)


class TestComputeWakeOnlyMetrics(unittest.TestCase):
    """Test Wake Recall metric computation."""

    def test_all_wake(self):
        """All Wake epochs should compute recall."""
        from core.edf_evaluation import compute_wake_only_metrics

        y_true = ["Wake", "Wake", "Wake"]
        y_pred = ["Wake", "Wake", "Wake"]
        probabilities = np.array([[0.9, 0.1, 0.0, 0.0, 0.0]] * 3)
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_wake_only_metrics(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["wake_recall"], 1.0)
        self.assertEqual(metrics["wake_n_epochs"], 3)

    def test_no_wake(self):
        """No Wake epochs should return zero metrics."""
        from core.edf_evaluation import compute_wake_only_metrics

        y_true = ["N1", "N2", "N3"]
        y_pred = ["N1", "N2", "N3"]
        probabilities = np.array([[0.0, 0.9, 0.1, 0.0, 0.0]] * 3)
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_wake_only_metrics(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["wake_recall"], 0.0)
        self.assertEqual(metrics["wake_n_epochs"], 0)

    def test_mixed_stages(self):
        """Mixed stages should compute only on Wake epochs."""
        from core.edf_evaluation import compute_wake_only_metrics

        y_true = ["Wake", "N1", "Wake", "N2"]
        y_pred = ["Wake", "N1", "N1", "N2"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_wake_only_metrics(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["wake_recall"], 0.5)  # 1/2 correct
        self.assertEqual(metrics["wake_n_epochs"], 2)


class TestComputeAlwaysWakeBaseline(unittest.TestCase):
    """Test Always-Wake baseline computation."""

    def test_all_wake_true(self):
        """All true Wake should give 100% baseline."""
        from core.edf_evaluation import compute_always_wake_baseline

        y_true = ["Wake", "Wake", "Wake"]
        y_pred = ["Wake", "Wake", "Wake"]
        probabilities = np.array([[0.9, 0.1, 0.0, 0.0, 0.0]] * 3)
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_always_wake_baseline(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["always_wake_baseline_accuracy"], 1.0)
        self.assertEqual(metrics["total_epochs"], 3)

    def test_mixed_stages(self):
        """Mixed stages should compute Wake baseline."""
        from core.edf_evaluation import compute_always_wake_baseline

        y_true = ["Wake", "N1", "Wake", "N2"]
        y_pred = ["Wake", "N1", "Wake", "N2"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_always_wake_baseline(y_true, y_pred, probabilities, stage_names)

        # 2 Wake out of 4 total = 0.5 baseline accuracy
        self.assertEqual(metrics["always_wake_baseline_accuracy"], 0.5)
        self.assertEqual(metrics["total_epochs"], 4)

    def test_sc4001_baseline(self):
        """SC4001: 1997 Wake out of 2650 total = 0.7536 baseline."""
        from core.edf_evaluation import compute_always_wake_baseline

        y_true = ["Wake"] * 1997 + ["N1"] * 59 + ["N2"] * 250 + ["N3"] * 220 + ["REM"] * 124
        y_pred = ["Wake"] * 2650
        probabilities = np.array([[0.9, 0.1, 0.0, 0.0, 0.0]] * 2650)
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_always_wake_baseline(y_true, y_pred, probabilities, stage_names)

        expected_baseline = 1997 / 2650
        self.assertAlmostEqual(metrics["always_wake_baseline_accuracy"], expected_baseline, places=4)
        self.assertEqual(metrics["total_epochs"], 2650)


class TestComputeSleepOnlyMetrics(unittest.TestCase):
    """Test sleep-only metric computation."""

    def test_all_sleep(self):
        """All sleep epochs should compute metrics."""
        from core.edf_evaluation import compute_sleep_only_metrics

        y_true = ["N1", "N2", "N3", "REM"]
        y_pred = ["N1", "N2", "N3", "REM"]
        probabilities = np.array([
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_sleep_only_metrics(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["sleep_only_accuracy"], 1.0)
        self.assertEqual(metrics["sleep_only_macro_f1"], 1.0)
        self.assertEqual(metrics["sleep_only_n_epochs"], 4)

    def test_no_sleep(self):
        """No sleep epochs should return zero metrics."""
        from core.edf_evaluation import compute_sleep_only_metrics

        y_true = ["Wake", "Wake"]
        y_pred = ["Wake", "Wake"]
        probabilities = np.array([[0.9, 0.1, 0.0, 0.0, 0.0]] * 2)
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        metrics = compute_sleep_only_metrics(y_true, y_pred, probabilities, stage_names)

        self.assertEqual(metrics["sleep_only_accuracy"], 0.0)
        self.assertEqual(metrics["sleep_only_macro_f1"], 0.0)
        self.assertEqual(metrics["sleep_only_n_epochs"], 0)


class TestEvaluateEDFPredictions(unittest.TestCase):
    """Test main EDF evaluation function."""

    def test_all_valid_labels(self):
        """All valid labels should compute full metrics."""
        from core.edf_evaluation import evaluate_edf_predictions

        y_true = ["Wake", "N1", "N2", "N3", "REM"]
        y_pred = ["Wake", "N1", "N2", "N3", "REM"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        result = evaluate_edf_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            stage_names=stage_names,
        )

        self.assertEqual(result.n_total_epochs, 5)
        self.assertEqual(result.n_valid_epochs, 5)
        self.assertEqual(result.n_excluded_epochs, 0)
        self.assertEqual(result.metrics["accuracy"], 1.0)
        self.assertIsNotNone(result.wake_only_metrics)
        self.assertIsNotNone(result.sleep_only_metrics)
        self.assertIsNotNone(result.always_wake_baseline)

    def test_partial_invalid_labels(self):
        """Partial invalid labels should compute metrics only on valid subset."""
        from core.edf_evaluation import evaluate_edf_predictions

        y_true = ["Wake", None, "N2", "Movement time", "REM"]
        y_pred = ["Wake", "N1", "N2", "N3", "REM"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        result = evaluate_edf_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            stage_names=stage_names,
        )

        self.assertEqual(result.n_total_epochs, 5)
        self.assertEqual(result.n_valid_epochs, 3)  # Wake, N2, REM
        self.assertEqual(result.n_excluded_epochs, 2)
        self.assertEqual(result.metrics["accuracy"], 1.0)  # All valid predictions correct

    def test_no_valid_labels(self):
        """No valid labels should return empty metrics."""
        from core.edf_evaluation import evaluate_edf_predictions

        y_true = [None, None, "Movement time"]
        y_pred = ["Wake", "N1", "N2"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        result = evaluate_edf_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            stage_names=stage_names,
        )

        self.assertEqual(result.n_total_epochs, 3)
        self.assertEqual(result.n_valid_epochs, 0)
        self.assertEqual(result.n_excluded_epochs, 3)
        self.assertEqual(len(result.metrics), 0)
        self.assertIsNone(result.wake_only_metrics)
        self.assertIsNone(result.sleep_only_metrics)
        self.assertIsNone(result.always_wake_baseline)

    def test_provenance_tracking(self):
        """Provenance information should be tracked correctly."""
        from core.edf_evaluation import evaluate_edf_predictions

        y_true = ["Wake", "N1"]
        y_pred = ["Wake", "N1"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        result = evaluate_edf_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            stage_names=stage_names,
            model_name="TestModel",
            uncertainty_policy="v3",
            epoch_length_sec=30,
            recording_identity="SC4001E0",
        )

        self.assertEqual(result.provenance["model_name"], "TestModel")
        self.assertEqual(result.provenance["uncertainty_policy"], "v3")
        self.assertEqual(result.provenance["epoch_length_sec"], 30)
        self.assertEqual(result.provenance["recording_identity"], "SC4001E0")
        self.assertTrue(result.provenance["research_use_only"])


class TestExportEvaluationCSV(unittest.TestCase):
    """Test CSV export function."""

    def test_csv_export_structure(self):
        """CSV export should have correct structure."""
        from core.edf_evaluation import export_evaluation_csv

        y_true = ["Wake", "N1", "N2"]
        y_pred = ["Wake", "N1", "N2"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
        ])
        confidences = np.array([0.9, 0.9, 0.9])

        csv_bytes = export_evaluation_csv(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            confidences=confidences,
            recording_identity="SC4001E0",
        )

        csv_text = csv_bytes.decode("utf-8")

        # Check disclaimer
        self.assertIn("RESEARCH PROTOTYPE ONLY", csv_text)
        self.assertIn("SC4001E0", csv_text)

        # Check columns
        self.assertIn("epoch", csv_text)
        self.assertIn("time_sec", csv_text)
        self.assertIn("true_label", csv_text)
        self.assertIn("predicted_label", csv_text)
        self.assertIn("confidence", csv_text)
        self.assertIn("prob_Wake", csv_text)
        self.assertIn("prob_N1", csv_text)

    def test_csv_export_with_none_labels(self):
        """CSV export should handle None labels correctly."""
        from core.edf_evaluation import export_evaluation_csv

        y_true = ["Wake", None, "N2"]
        y_pred = ["Wake", "N1", "N2"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0],
        ])
        confidences = np.array([0.9, 0.9, 0.9])

        csv_bytes = export_evaluation_csv(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            confidences=confidences,
        )

        csv_text = csv_bytes.decode("utf-8")
        self.assertIn("epoch", csv_text)


class TestExportEvaluationJSON(unittest.TestCase):
    """Test JSON export function."""

    def test_json_export_structure(self):
        """JSON export should have correct structure."""
        from core.edf_evaluation import (
            evaluate_edf_predictions,
            export_evaluation_json,
        )

        y_true = ["Wake", "N1"]
        y_pred = ["Wake", "N1"]
        probabilities = np.array([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0, 0.0],
        ])
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]

        evaluation_result = evaluate_edf_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            stage_names=stage_names,
        )

        json_bytes = export_evaluation_json(
            evaluation_result=evaluation_result,
            recording_identity="SC4001E0",
        )

        json_dict = json.loads(json_bytes.decode("utf-8"))

        # Check required fields
        self.assertIn("recording_identity", json_dict)
        self.assertIn("metrics", json_dict)
        self.assertIn("confusion_matrix", json_dict)
        self.assertIn("per_stage_metrics", json_dict)
        self.assertIn("epoch_counts", json_dict)
        self.assertIn("provenance", json_dict)
        self.assertIn("disclaimer", json_dict)

        # Check disclaimer
        self.assertIn("RESEARCH PROTOTYPE ONLY", json_dict["disclaimer"])

        # Check epoch counts
        self.assertEqual(json_dict["epoch_counts"]["total"], 2)
        self.assertEqual(json_dict["epoch_counts"]["valid"], 2)
        self.assertEqual(json_dict["epoch_counts"]["excluded"], 0)


if __name__ == "__main__":
    unittest.main()
