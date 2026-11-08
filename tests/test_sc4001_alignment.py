"""
Investigate SC4001 annotation alignment.

PSG duration: 79500 s
Hypnogram annotation extent: 86400 s
All PSG epochs: 2650 valid labels

This test checks:
- Whether excess annotation duration is only a trailing overhang
- Shared time origin and onset alignment
- Annotation coverage of the PSG
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


class TestSC4001Alignment(unittest.TestCase):
    """Verify SC4001 annotation alignment with PSG."""

    def setUp(self):
        self.psg_path, self.hypnogram_path = find_sc4001_files()
        if self.psg_path is None or self.hypnogram_path is None:
            self.skipTest("Real SC4001 EDF recordings not present on filesystem.")

    def test_sc4001_alignment(self):
        """Verify annotations start at t=0 and fully cover PSG."""
        raw = mne.io.read_raw_edf(self.psg_path, preload=False, verbose="ERROR")
        psg_duration_sec = float(raw.n_times / raw.info["sfreq"])
        annotations = mne.read_annotations(self.hypnogram_path)

        onsets = np.asarray(annotations.onset, dtype=float)
        durations = np.asarray(annotations.duration, dtype=float)

        first_onset = float(onsets.min())
        last_end = float((onsets + durations).max())
        annotation_coverage_end = min(last_end, psg_duration_sec)
        annotation_coverage_duration = annotation_coverage_end - first_onset

        # Verify time origin is zero (within 1 second)
        self.assertAlmostEqual(first_onset, 0.0, delta=1.0, msg="Annotations must start at t=0")

        # Verify trailing overhang
        self.assertTrue(last_end >= psg_duration_sec, "Hypnogram must cover full PSG duration")

        # Verify coverage is >= 99% of PSG duration
        coverage_pct = annotation_coverage_duration / psg_duration_sec
        self.assertGreaterEqual(coverage_pct, 0.99, "Annotations must cover at least 99% of PSG")


if __name__ == "__main__":
    unittest.main()
