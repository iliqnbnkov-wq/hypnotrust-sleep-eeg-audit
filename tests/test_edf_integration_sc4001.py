"""
Integration test for EDF evaluation with SC4001 (Sleep-EDF).

This test verifies that the EDF pipeline produces the expected metrics
when evaluated against the verified SC4001 hypnogram.

Reference values (from frozen V5d evaluation):
- Accuracy: 0.9702
- Balanced Accuracy: 0.8340
- Macro F1: 0.8407
- Cohen's Kappa: 0.9276
- Sleep-only Accuracy: 0.8821
- Sleep-only Macro F1: 0.8039

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_sc4001_files() -> tuple[Path | None, Path | None]:
    """Locate real SC4001 PSG and hypnogram without committing them to the repository."""
    candidates = [
        PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data",
        Path(os.environ.get(
            "HYPNOTRUST_EDF_DIR",
            r"C:\Users\User\Documents\Uni\egg_project\test_files\sleep_edf_real\physionet-sleep-data"
        )),
        Path(r"C:\Users\User\Documents\Uni\egg_project\test_files\sleep_edf_real\physionet-sleep-data"),
    ]
    for d in candidates:
        p = d / "SC4001E0-PSG.edf"
        h = d / "SC4001EC-Hypnogram.edf"
        if p.exists() and h.exists():
            return p, h
    return None, None


class TestEDFIntegrationSC4001(unittest.TestCase):
    """End-to-end integration test for EDF pipeline on SC4001."""

    def setUp(self):
        self.psg_path, self.hypnogram_path = find_sc4001_files()
        if self.psg_path is None or self.hypnogram_path is None:
            self.skipTest("Real SC4001 EDF recordings not present on filesystem.")

    def test_sc4001_edf_evaluation(self):
        """Run full EDF pipeline on SC4001 and verify metrics against frozen V5d reference."""
        raw = mne.io.read_raw_edf(self.psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(self.hypnogram_path)

        from core.config import get_default_pipeline_config
        from core.pipeline import run_full_pipeline
        from core.edf_evaluation import evaluate_edf_predictions

        config = get_default_pipeline_config()

        result = run_full_pipeline(
            raw=raw,
            annotations=annotations,
            config=config,
            subject=4001,
            recording=0,
        )

        evaluation_result = evaluate_edf_predictions(
            y_true=result.labels,
            y_pred=result.predicted_stages,
            probabilities=result.probabilities,
            stage_names=result.stage_names,
            raw_duration_sec=float(raw.n_times / raw.info["sfreq"]),
            annotations_provided=True,
            model_name="Temporal v2 RandomForest",
            uncertainty_policy=config.uncertainty_policy,
            epoch_length_sec=config.epoch_length_sec,
            recording_identity="SC4001E0",
        )

        metrics = evaluation_result.metrics
        TOLERANCE = 0.01  # 1% tolerance

        self.assertAlmostEqual(metrics["accuracy"], 0.9702, delta=TOLERANCE)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.8340, delta=TOLERANCE)
        self.assertAlmostEqual(metrics["macro_f1"], 0.8407, delta=TOLERANCE)
        self.assertAlmostEqual(metrics["cohen_kappa"], 0.9276, delta=TOLERANCE)

        self.assertIsNotNone(evaluation_result.sleep_only_metrics)
        self.assertAlmostEqual(evaluation_result.sleep_only_metrics["sleep_only_accuracy"], 0.8821, delta=TOLERANCE)
        self.assertAlmostEqual(evaluation_result.sleep_only_metrics["sleep_only_macro_f1"], 0.8039, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
