"""
EEG Waveform Extraction Module for HypnoTrust v2.

Extracts raw EEG signal segments around transition events from EDF files.

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np


@dataclass(frozen=True)
class EEGSegment:
    """
    Represents an extracted EEG signal segment.

    Attributes:
        channel: Channel name (e.g., "Fpz-Cz")
        signal_uv: Signal in microvolts
        time_s: Time axis in seconds relative to event boundary
        sfreq: Sampling frequency in Hz
        event_boundary_sample: Sample index of the boundary in the original raw signal
    """

    channel: str
    signal_uv: np.ndarray
    time_s: np.ndarray
    sfreq: float
    event_boundary_sample: int


def extract_eeg_segment(
    raw: mne.io.BaseRaw,
    channel: str,
    event_boundary_sample: int,
    event_window_s: tuple[float, float] = (-60.0, 60.0),
) -> EEGSegment:
    """
    Extract EEG signal segment around an event boundary.

    Args:
        raw: MNE Raw object
        channel: Channel name to extract
        event_boundary_sample: Sample index of the event boundary
        event_window_s: Event window relative to boundary (start, end) in seconds

    Returns:
        EEGSegment with signal and time axis
    """
    sfreq = float(raw.info["sfreq"])
    window_start_s, window_end_s = event_window_s

    window_start_sample = int(event_boundary_sample + window_start_s * sfreq)
    window_end_sample = int(event_boundary_sample + window_end_s * sfreq)

    # Resolve channel name - inline for now
    actual_channel = None
    for ch in raw.ch_names:
        if channel.lower() in ch.lower():
            actual_channel = ch
            break

    if actual_channel is None:
        raise ValueError(f"Channel {channel} not found in EDF")

    # Extract signal
    signal_si = raw.get_data(picks=[actual_channel])[0]
    signal_segment_si = signal_si[window_start_sample:window_end_sample]

    # Convert to microvolts
    signal_uv = signal_segment_si * 1e6

    # Create time axis relative to boundary
    n_samples = len(signal_uv)
    time_s = np.arange(n_samples) / sfreq + window_start_s

    return EEGSegment(
        channel=channel,
        signal_uv=signal_uv,
        time_s=time_s,
        sfreq=sfreq,
        event_boundary_sample=event_boundary_sample,
    )
