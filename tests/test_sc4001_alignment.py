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

import sys
from pathlib import Path

import mne

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def investigate_sc4001_alignment():
    """Investigate SC4001 annotation alignment."""
    psg_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001E0-PSG.edf"
    hypnogram_path = PROJECT_ROOT / "test_files" / "sleep_edf_real" / "physionet-sleep-data" / "SC4001EC-Hypnogram.edf"

    if not psg_path.exists() or not hypnogram_path.exists():
        print("SKIP: SC4001 files not found")
        return

    print("=" * 70)
    print("SC4001 ANNOTATION ALIGNMENT INVESTIGATION")
    print("=" * 70)

    # Load PSG
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    psg_duration_sec = float(raw.n_times / raw.info["sfreq"])
    print(f"\nPSG Duration: {psg_duration_sec:.1f} seconds ({psg_duration_sec / 3600:.2f} hours)")
    print(f"PSG Samples: {raw.n_times}")
    print(f"PSG Sampling Rate: {raw.info['sfreq']} Hz")

    # Load hypnogram
    annotations = mne.read_annotations(hypnogram_path)
    print(f"\nHypnogram Annotations: {len(annotations)} segments")

    # Calculate annotation extent
    onsets = np.asarray(annotations.onset, dtype=float)
    durations = np.asarray(annotations.duration, dtype=float)

    first_onset = float(onsets.min())
    last_onset = float(onsets.max())
    last_end = float((onsets + durations).max())
    annotation_extent = last_end - first_onset

    print(f"\nAnnotation First Onset: {first_onset:.1f} seconds")
    print(f"Annotation Last Onset: {last_onset:.1f} seconds")
    print(f"Annotation Last End: {last_end:.1f} seconds")
    print(f"Annotation Extent: {annotation_extent:.1f} seconds ({annotation_extent / 3600:.2f} hours)")

    # Check alignment
    print(f"\nPSG vs Annotation Duration:")
    print(f"  PSG: {psg_duration_sec:.1f} s")
    print(f"  Annotation extent: {annotation_extent:.1f} s")
    print(f"  Difference: {annotation_extent - psg_duration_sec:.1f} s")

    # Check if excess is trailing overhang
    excess_duration = annotation_extent - psg_duration_sec
    trailing_overhang = last_end > psg_duration_sec

    print(f"\nExcess Analysis:")
    print(f"  Excess duration: {excess_duration:.1f} s")
    print(f"  Is trailing overhang: {trailing_overhang}")

    if trailing_overhang:
        print(f"  Trailing overhang: {last_end - psg_duration_sec:.1f} s")
        print(f"  This represents: {(last_end - psg_duration_sec) / 3600:.2f} hours")

    # Check annotation coverage of PSG
    annotation_coverage_start = first_onset
    annotation_coverage_end = min(last_end, psg_duration_sec)
    annotation_coverage_duration = annotation_coverage_end - annotation_coverage_start

    print(f"\nAnnotation Coverage of PSG:")
    print(f"  Coverage start: {annotation_coverage_start:.1f} s")
    print(f"  Coverage end: {annotation_coverage_end:.1f} s")
    print(f"  Coverage duration: {annotation_coverage_duration:.1f} s")
    print(f"  Coverage percentage: {annotation_coverage_duration / psg_duration_sec * 100:.1f}%")

    # Check if annotations start at time 0
    starts_at_zero = abs(first_onset) < 1.0  # 1 second tolerance
    print(f"\nTime Origin:")
    print(f"  Annotations start at t=0: {starts_at_zero}")
    print(f"  First onset: {first_onset:.1f} s")

    # Determine alignment verdict
    print(f"\n" + "=" * 70)
    print("ALIGNMENT VERDICT")
    print("=" * 70)

    if starts_at_zero and trailing_overhang and (annotation_coverage_duration >= psg_duration_sec * 0.99):
        print("VALID: Annotations start at t=0 and fully cover PSG with trailing overhang.")
        print("The excess annotation duration is a trailing overhang beyond PSG end.")
        print("This is acceptable for Sleep-EDF annotation format.")
        return True
    elif starts_at_zero and abs(annotation_extent - psg_duration_sec) < 120:  # 2 minutes tolerance
        print("VALID: Annotations align with PSG within small tolerance.")
        return True
    else:
        print("WARNING: Annotation alignment may be problematic.")
        if not starts_at_zero:
            print("  - Annotations do not start at t=0")
        if not trailing_overhang:
            print("  - Excess is not a trailing overhang")
        if annotation_coverage_duration < psg_duration_sec * 0.99:
            print("  - Annotations do not fully cover PSG")
        return False


if __name__ == "__main__":
    import numpy as np
    success = investigate_sc4001_alignment()
    sys.exit(0 if success else 1)
