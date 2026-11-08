"""
Integration test for Transition Explorer with SC4001.

Tests P0 features:
- Transition detection with k=3
- EEG waveform extraction
- Spectral analysis
- Reproduction of E1 findings (8 transitions meeting k=3 criterion: 1 N2->N3, 7 N3->N2)
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd

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


class TestTransitionIntegrationSC4001(unittest.TestCase):
    """End-to-end integration tests for Transition Explorer on real SC4001 data."""

    def setUp(self):
        self.psg_path, self.hypnogram_path = find_sc4001_files()
        if self.psg_path is None or self.hypnogram_path is None:
            self.skipTest("Real SC4001 EDF recordings not present on filesystem.")

    def test_sc4001_transition_detection_and_e1_counts(self):
        """Test transition detection with k=3 reproduces E1 counts exactly."""
        raw = mne.io.read_raw_edf(self.psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(self.hypnogram_path)

        from core.feature_extraction import _labels_from_annotations
        from core.transition_detection import detect_transitions, count_transitions_by_kind

        n_epochs = int(raw.n_times / raw.info["sfreq"] // 30)
        labels = _labels_from_annotations(annotations, n_epochs, 30.0)
        self.assertIsNotNone(labels)

        # Detect transitions with k=3 stability criterion
        events_k3 = detect_transitions(labels, 30.0, k=3)
        counts_k3 = count_transitions_by_kind(events_k3)

        # E1 reference benchmark: 8 transitions (1 N2->N3, 7 N3->N2)
        self.assertEqual(counts_k3["N2->N3"], 1, "Expected exactly 1 N2->N3 transition")
        self.assertEqual(counts_k3["N3->N2"], 7, "Expected exactly 7 N3->N2 transitions")
        self.assertEqual(counts_k3["BOTH"], 8, "Expected exactly 8 total transitions")

        # Verify exact boundary seconds match E1 pilot
        expected_boundaries = [32250.0, 35790.0, 38760.0, 38910.0, 43710.0, 44520.0, 48300.0, 48780.0]
        actual_boundaries = [e.boundary_seconds for e in events_k3]
        self.assertEqual(actual_boundaries, expected_boundaries)

    def test_sc4001_spectral_analysis_matches_e1(self):
        """Test spectral analysis on SC4001 matches E1 reference values within numerical precision."""
        raw = mne.io.read_raw_edf(self.psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(self.hypnogram_path)

        from core.feature_extraction import _labels_from_annotations
        from core.transition_detection import detect_transitions
        from core.eeg_extraction import extract_eeg_segment
        from core.spectral_analysis import compute_spectral_windows, compute_before_after_power

        n_epochs = int(raw.n_times / raw.info["sfreq"] // 30)
        labels = _labels_from_annotations(annotations, n_epochs, 30.0)
        events_k3 = detect_transitions(labels, 30.0, k=3)
        self.assertTrue(len(events_k3) >= 1)

        # Test event 0 (boundary at 32250.0s, N3->N2)
        ev0 = events_k3[0]
        self.assertEqual(ev0.boundary_seconds, 32250.0)
        self.assertEqual(ev0.kind, "N3->N2")

        b_sample = int(ev0.boundary_seconds * raw.info["sfreq"])
        fpz_seg = extract_eeg_segment(raw, "Fpz-Cz", b_sample, (-60.0, 60.0))
        windows = compute_spectral_windows(fpz_seg.signal_uv, fpz_seg.sfreq)

        self.assertEqual(len(windows), 23, "Expected 23 sliding windows (-55s to +55s)")
        self.assertAlmostEqual(windows[0].relative_center_s, -55.0, places=4)
        self.assertAlmostEqual(windows[-1].relative_center_s, 55.0, places=4)

        # Compare with original E1 reference values for 32250s
        # First window power_uv2 ~ 741.815, power_db ~ 28.703
        self.assertAlmostEqual(windows[0].power_uv2, 741.815, places=2)
        self.assertAlmostEqual(windows[0].power_db_uv2, 28.703, places=2)

        # Check before/after delta power
        ba = compute_before_after_power(windows)
        self.assertAlmostEqual(ba["before_power_db"], 28.21, places=1)
        self.assertAlmostEqual(ba["after_power_db"], 23.98, places=1)
        self.assertTrue(ba["power_change_db"] < 0, "Slow-wave power must drop from N3 to N2")

    def test_sc4001_all_368_spectral_windows_reproduce_e1_csv(self):
        """Verify all 368 spectral windows match frozen results/pilot_e1/e1_spectral_windows.csv."""
        e1_csv_path = PROJECT_ROOT / "results" / "pilot_e1" / "e1_spectral_windows.csv"
        if not e1_csv_path.exists():
            self.skipTest("Frozen reference results/pilot_e1/e1_spectral_windows.csv not found")

        e1_df = pd.read_csv(e1_csv_path)
        self.assertEqual(len(e1_df), 368, "Expected 368 rows (8 events * 2 channels * 23 windows)")

        raw = mne.io.read_raw_edf(self.psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(self.hypnogram_path)

        from core.feature_extraction import _labels_from_annotations
        from core.transition_detection import detect_transitions
        from core.eeg_extraction import extract_eeg_segment
        from core.spectral_analysis import compute_spectral_windows

        n_epochs = int(raw.n_times / raw.info["sfreq"] // 30)
        labels = _labels_from_annotations(annotations, n_epochs, 30.0)
        events_k3 = detect_transitions(labels, 30.0, k=3)
        self.assertEqual(len(events_k3), 8)

        computed_rows = []
        for ev in events_k3:
            b_sample = int(ev.boundary_seconds * raw.info["sfreq"])
            for ch in ["Fpz-Cz", "Pz-Oz"]:
                ch_name = f"EEG {ch}"
                seg = extract_eeg_segment(raw, ch, b_sample, (-60.0, 60.0))
                windows = compute_spectral_windows(seg.signal_uv, seg.sfreq)
                for w in windows:
                    computed_rows.append({
                        "channel": ch_name,
                        "boundary_seconds": ev.boundary_seconds,
                        "relative_center_s": w.relative_center_s,
                        "power_uv2": w.power_uv2,
                        "power_db_uv2": w.power_db_uv2,
                    })

        comp_df = pd.DataFrame(computed_rows)
        self.assertEqual(len(comp_df), 368)

        # Merge and compare
        merged = pd.merge(
            comp_df,
            e1_df,
            on=["channel", "boundary_seconds", "relative_center_s"],
            suffixes=("_calc", "_ref"),
        )
        self.assertEqual(len(merged), 368)

        max_power_diff = np.max(np.abs(merged["power_uv2_calc"] - merged["power_uv2_ref"]))
        max_db_diff = np.max(np.abs(merged["power_db_uv2_calc"] - merged["power_db_uv2_ref"]))

        self.assertLess(max_power_diff, 1e-6, f"Max power diff {max_power_diff} exceeds tolerance")
        self.assertLess(max_db_diff, 1e-6, f"Max dB diff {max_db_diff} exceeds tolerance")


if __name__ == "__main__":
    unittest.main()
