"""
Spectral Analysis Module for HypnoTrust v2.

Computes slow-wave (0.5-2 Hz) bandpower from EEG signals using Welch PSD.

Based on E1 methodology:
- Welch PSD estimation
- 4-second segments
- 50% overlap
- 10-second sliding windows
- 5-second sliding step
- Event-centered time axis

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class SpectralWindow:
    """
    Represents a single spectral analysis window.

    Attributes:
        relative_center_s: Time center relative to event boundary (seconds)
        stage_relative: "N2_or_N3_before", "mixed_at_boundary", or "N2_or_N3_after"
        power_uv2: Power in µV^2
        power_db_uv2: Power in dB (10*log10(power / 1µV^2))
        qc: Quality control status ("ok", "amplitude_over_limit", "near_flatline", "nonfinite")
    """

    relative_center_s: float
    stage_relative: str
    power_uv2: float
    power_db_uv2: float
    qc: str


def compute_spectral_windows(
    signal_uv: np.ndarray,
    sfreq: float,
    event_boundary_sample: int | None = None,
    event_window_s: tuple[float, float] = (-60.0, 60.0),
    psd_window_s: float = 10.0,
    window_step_s: float = 5.0,
    welch_segment_s: float = 4.0,
    welch_overlap: float = 0.5,
    band_hz: tuple[float, float] = (0.5, 2.0),
    qc_peak_to_peak_uv: float = 500.0,
    qc_min_std_uv: float = 0.5,
) -> list[SpectralWindow]:
    """
    Compute sliding-window spectral power around an event boundary.

    Args:
        signal_uv: EEG signal in microvolts. Can be either:
                   1) A pre-extracted segment spanning event_window_s (e.g. from extract_eeg_segment)
                   2) Full continuous recording channel (requires event_boundary_sample)
        sfreq: Sampling frequency in Hz
        event_boundary_sample: Sample index of the event boundary in the signal array (optional if pre-extracted segment)
        event_window_s: Event window relative to boundary (start, end) in seconds
        psd_window_s: PSD window duration in seconds
        window_step_s: Sliding window step in seconds
        welch_segment_s: Welch segment duration in seconds
        welch_overlap: Welch segment overlap (0-1)
        band_hz: Frequency band for power calculation (low, high) in Hz
        qc_peak_to_peak_uv: QC threshold for peak-to-peak amplitude
        qc_min_std_uv: QC threshold for minimum standard deviation

    Returns:
        List of SpectralWindow objects
    """
    window_start_s, window_end_s = event_window_s
    expected_seg_samples = int(round((window_end_s - window_start_s) * sfreq))

    # Detect if signal_uv is already the extracted segment around event
    if len(signal_uv) <= expected_seg_samples + int(round(sfreq)):
        segment = signal_uv
    else:
        if event_boundary_sample is None:
            raise ValueError("event_boundary_sample must be provided when passing continuous full-recording signal.")
        start_sample = int(round(event_boundary_sample + window_start_s * sfreq))
        stop_sample = int(round(event_boundary_sample + window_end_s * sfreq))
        if start_sample < 0 or stop_sample > len(signal_uv):
            return []
        segment = signal_uv[start_sample:stop_sample]

    if len(segment) == 0:
        return []

    # Generate sliding windows
    window_samples = int(round(psd_window_s * sfreq))
    step_samples = int(round(window_step_s * sfreq))

    if len(segment) < window_samples:
        return []

    n_windows = int((len(segment) - window_samples) // step_samples) + 1
    windows: list[SpectralWindow] = []

    seg_len_samples = int(round(welch_segment_s * sfreq))
    noverlap_samples = int(round(seg_len_samples * welch_overlap))

    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        window_signal = segment[start_idx:end_idx]

        # Quality control
        if not np.isfinite(window_signal).all():
            qc = "nonfinite"
        elif float(np.ptp(window_signal)) > qc_peak_to_peak_uv:
            qc = "amplitude_over_limit"
        elif float(np.std(window_signal)) < qc_min_std_uv:
            qc = "near_flatline"
        else:
            qc = "ok"

        # Compute Welch PSD (Hann window, constant detrend, density scaling)
        freqs, psd = welch(
            window_signal,
            fs=sfreq,
            window="hann",
            nperseg=seg_len_samples,
            noverlap=noverlap_samples,
            detrend="constant",
            scaling="density",
        )

        # Extract band power (0.5 to 2.0 Hz inclusive)
        band_mask = (freqs >= band_hz[0] - 1e-10) & (freqs <= band_hz[1] + 1e-10)
        band_power = float(np.trapezoid(psd[band_mask], freqs[band_mask]))

        # Convert to dB: 10*log10(power / 1µV^2)
        power_db = float(10.0 * np.log10(band_power)) if band_power > 0 else float("nan")

        # Relative time from boundary (center of current window relative to boundary t=0)
        relative_center_s = float(window_start_s + (start_idx + window_samples / 2.0) / sfreq)
        if relative_center_s < 0:
            stage_rel = "N2_or_N3_before"
        elif abs(relative_center_s) < 1e-6:
            stage_rel = "mixed_at_boundary"
        else:
            stage_rel = "N2_or_N3_after"

        windows.append(
            SpectralWindow(
                relative_center_s=relative_center_s,
                stage_relative=stage_rel,
                power_uv2=band_power,
                power_db_uv2=power_db,
                qc=qc,
            )
        )

    return windows


def compute_before_after_power(
    windows: list[SpectralWindow],
    before_threshold_s: float = -5.0,
    after_threshold_s: float = 5.0,
) -> dict[str, float]:
    """
    Compute average power before and after the event boundary.

    Args:
        windows: List of SpectralWindow objects
        before_threshold_s: Time threshold for "before" classification (seconds)
        after_threshold_s: Time threshold for "after" classification (seconds)

    Returns:
        Dictionary with before/after power in dB and dB change
    """
    before_windows = [
        w for w in windows
        if w.relative_center_s < before_threshold_s and w.qc == "ok" and np.isfinite(w.power_db_uv2)
    ]
    after_windows = [
        w for w in windows
        if w.relative_center_s > after_threshold_s and w.qc == "ok" and np.isfinite(w.power_db_uv2)
    ]

    if not before_windows or not after_windows:
        return {
            "before_power_db": float("nan"),
            "after_power_db": float("nan"),
            "power_change_db": float("nan"),
            "n_before": len(before_windows),
            "n_after": len(after_windows),
        }

    before_power_db = float(np.mean([w.power_db_uv2 for w in before_windows]))
    after_power_db = float(np.mean([w.power_db_uv2 for w in after_windows]))
    power_change_db = after_power_db - before_power_db

    return {
        "before_power_db": before_power_db,
        "after_power_db": after_power_db,
        "power_change_db": power_change_db,
        "n_before": len(before_windows),
        "n_after": len(after_windows),
    }
