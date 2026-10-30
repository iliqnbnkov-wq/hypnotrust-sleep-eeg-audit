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
        stage_relative: "before", "after", or "mixed"
        power_uv2: Power in µV^2
        power_db_uv2: Power in dB (10*log10(power / 1µV^2))
        qc: Quality control status
    """

    relative_center_s: float
    stage_relative: str
    power_uv2: float
    power_db_uv2: float
    qc: str


def compute_spectral_windows(
    signal_uv: np.ndarray,
    sfreq: float,
    event_boundary_sample: int,
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
        signal_uv: EEG signal in microvolts (already extracted around event)
        sfreq: Sampling frequency in Hz
        event_boundary_sample: Sample index of the event boundary IN THE EXTRACTED SIGNAL
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
    window_start_sample = int(event_boundary_sample + window_start_s * sfreq)
    window_end_sample = int(event_boundary_sample + window_end_s * sfreq)

    # Extract signal segment
    segment = signal_uv[window_start_sample:window_end_sample]

    if len(segment) == 0:
        return []

    # Generate sliding windows
    window_samples = int(psd_window_s * sfreq)
    step_samples = int(window_step_s * sfreq)

    windows = []
    n_windows = (len(segment) - window_samples) // step_samples

    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        window_signal = segment[start_idx:end_idx]

        # Quality control
        peak_to_peak = np.max(window_signal) - np.min(window_signal)
        std_dev = np.std(window_signal)

        if peak_to_peak > qc_peak_to_peak_uv:
            qc = "high_amplitude"
        elif std_dev < qc_min_std_uv:
            qc = "low_variance"
        else:
            qc = "ok"

        # Compute Welch PSD
        freqs, psd = welch(
            window_signal,
            fs=sfreq,
            nperseg=int(welch_segment_s * sfreq),
            noverlap=int(welch_segment_s * welch_overlap * sfreq),
        )

        # Extract band power
        band_mask = (freqs >= band_hz[0]) & (freqs < band_hz[1])
        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])

        # Convert to dB: 10*log10(power / 1µV^2)
        power_db = 10 * np.log10(band_power) if band_power > 0 else -np.inf

        # Relative time from boundary
        window_center_sample = start_idx + window_samples // 2
        absolute_center_sample = window_start_sample + window_center_sample
        relative_center_s = (absolute_center_sample - event_boundary_sample) / sfreq

        windows.append(
            SpectralWindow(
                relative_center_s=relative_center_s,
                stage_relative="unknown",  # Will be set by caller based on hypnogram
                power_uv2=float(band_power),
                power_db_uv2=float(power_db),
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
    before_windows = [w for w in windows if w.relative_center_s < before_threshold_s]
    after_windows = [w for w in windows if w.relative_center_s > after_threshold_s]

    if not before_windows or not after_windows:
        return {
            "before_power_db": np.nan,
            "after_power_db": np.nan,
            "power_change_db": np.nan,
            "n_before": len(before_windows),
            "n_after": len(after_windows),
        }

    before_power_db = np.mean([w.power_db_uv2 for w in before_windows])
    after_power_db = np.mean([w.power_db_uv2 for w in after_windows])
    power_change_db = after_power_db - before_power_db

    return {
        "before_power_db": float(before_power_db),
        "after_power_db": float(after_power_db),
        "power_change_db": float(power_change_db),
        "n_before": len(before_windows),
        "n_after": len(after_windows),
    }
