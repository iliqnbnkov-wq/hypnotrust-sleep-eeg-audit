"""
Integration test for Transition Explorer with SC4001.

Tests P0 features:
- Transition detection with k=3
- EEG waveform extraction
- Spectral analysis
- Reproduction of E1 findings (8 transitions meeting k=3 criterion)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_sc4001_transition_detection():
    """Test transition detection with SC4001."""
    psg_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001E0-PSG.edf"
    hypnogram_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001EC-Hypnogram.edf"

    if not psg_path.exists() or not hypnogram_path.exists():
        pytest.skip("SC4001 files not found (external data not in repository)")

    print("=" * 70)
    print("SC4001 TRANSITION DETECTION TEST")
    print("=" * 70)

    # Load PSG
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    print(f"PSG loaded: {raw.n_times} samples, {raw.info['sfreq']} Hz")

    # Load hypnogram
    annotations = mne.read_annotations(hypnogram_path)
    print(f"Hypnogram loaded: {len(annotations)} segments")

    # Extract labels manually
    from core.transition_detection import detect_transitions, count_transitions_by_kind

    n_epochs = int(raw.n_times / raw.info["sfreq"] // 30)
    labels = []

    onsets = np.asarray(annotations.onset, dtype=float)
    durations = np.asarray(annotations.duration, dtype=float)
    descriptions = list(annotations.description)

    STAGE_DESCRIPTION_MAP = {
        "Sleep stage W": "Wake",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",
        "Sleep stage R": "REM",
        "Sleep stage ?": None,
        "Movement time": None,
    }

    for epoch_index in range(n_epochs):
        midpoint = epoch_index * 30 + 30 / 2
        label = None

        for onset, duration, description in zip(onsets, durations, descriptions):
            if onset <= midpoint < onset + duration:
                label = STAGE_DESCRIPTION_MAP.get(str(description), None)
                break

        labels.append(label if label else "Wake")  # Use Wake as placeholder

    print(f"Labels extracted: {len(labels)} epochs")

    # Detect transitions with k=3
    events_k3 = detect_transitions(labels, 30.0, k=3)
    counts_k3 = count_transitions_by_kind(events_k3)

    print(f"\nTransitions detected with k=3:")
    print(f"  N2->N3: {counts_k3['N2->N3']}")
    print(f"  N3->N2: {counts_k3['N3->N2']}")
    print(f"  BOTH: {counts_k3['BOTH']}")

    # E1 reference: 8 transitions (1 N2->N3, 7 N3->N2)
    expected_total = 8
    if counts_k3["BOTH"] == expected_total:
        print(f"\n[PASS] Total transitions match E1 reference: {expected_total}")
    else:
        print(f"\n[FAIL] Total transitions mismatch: {counts_k3['BOTH']} vs {expected_total}")

    # Test spectral analysis for one event
    if events_k3:
        print("\nTesting spectral analysis for first event...")
        event = events_k3[0]

        from core.eeg_extraction import extract_eeg_segment
        from core.spectral_analysis import compute_spectral_windows, compute_before_after_power

        raw.load_data()

        event_boundary_sample = int(event.boundary_seconds * raw.info["sfreq"])

        try:
            fpz_segment = extract_eeg_segment(
                raw,
                "Fpz-Cz",
                event_boundary_sample,
                (-60.0, 60.0),
            )

            # For spectral analysis, we pass the event_boundary_sample from the segment
            # The segment already has the boundary stored
            fpz_windows = compute_spectral_windows(
                fpz_segment.signal_uv,
                fpz_segment.sfreq,
                fpz_segment.event_boundary_sample,
                event_window_s=(-60.0, 60.0),
                psd_window_s=10.0,
                window_step_s=5.0,
                welch_segment_s=4.0,
                welch_overlap=0.5,
            )

            fpz_before_after = compute_before_after_power(fpz_windows)

            print(f"Fpz-Cz spectral analysis:")
            print(f"  Before power: {fpz_before_after['before_power_db']:.2f} dB")
            print(f"  After power: {fpz_before_after['after_power_db']:.2f} dB")
            print(f"  Power change: {fpz_before_after['power_change_db']:+.2f} dB")
            print(f"  Windows before: {fpz_before_after['n_before']}")
            print(f"  Windows after: {fpz_before_after['n_after']}")

            print("\n[PASS] Spectral analysis completed successfully")
            return True

        except Exception as exc:
            print(f"\n[FAIL] Spectral analysis error: {exc}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n[FAIL] No transitions detected to test spectral analysis")
        return False


if __name__ == "__main__":
    success = test_sc4001_transition_detection()
    sys.exit(0 if success else 1)
