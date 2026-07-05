from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from hypnotrust.constants import BASELINE_FEATURE_COLUMNS
from hypnotrust.temporal import build_temporal_feature_dataframe


EEG_CHANNELS = ("Fpz-Cz", "Pz-Oz")
EOG_CHANNEL = "horizontal"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Sigma": (12.0, 16.0),
    "Beta": (16.0, 30.0),
}

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


@dataclass(frozen=True)
class FeatureExtractionResult:
    """
    Result of EDF feature extraction.

    baseline_df:
        Subject/Recording/Epoch/Stage + 13 baseline v1 features.

    temporal_df:
        Subject/Recording/Epoch/Stage + 65 temporal v2 features.

    labels:
        Ground-truth stage labels if annotations were provided and usable.
        None otherwise.
    """

    baseline_df: pd.DataFrame
    temporal_df: pd.DataFrame
    labels: list[str] | None


def _resolve_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    """
    Resolve a canonical Sleep-EDF channel name against MNE channel names.

    MNE with infer_types=True often converts:
    - EEG Fpz-Cz -> Fpz-Cz
    - EEG Pz-Oz -> Pz-Oz
    - EOG horizontal -> horizontal
    """

    if requested in raw.ch_names:
        return requested

    aliases = {
        "Fpz-Cz": ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ", "FPZ-CZ"],
        "Pz-Oz": ["EEG Pz-Oz", "Pz-Oz", "EEG PZ-OZ", "PZ-OZ"],
        "horizontal": ["EOG horizontal", "horizontal", "EOG Horizontal", "EOG"],
    }

    for alias in aliases.get(requested, []):
        if alias in raw.ch_names:
            return alias

    normalized = {
        ch.lower().replace("eeg", "").replace("eog", "").replace(" ", "").replace("-", ""): ch
        for ch in raw.ch_names
    }

    key = requested.lower().replace("eeg", "").replace("eog", "").replace(" ", "").replace("-", "")

    if key in normalized:
        return normalized[key]

    raise ValueError(
        f"Required channel '{requested}' was not found. "
        f"Available channels: {raw.ch_names}"
    )


def _epoch_signal(signal: np.ndarray, samples_per_epoch: int) -> np.ndarray:
    """
    Cut a 1D signal into full 30-second epochs.

    Partial trailing epochs are dropped.
    """

    n_epochs = len(signal) // samples_per_epoch

    if n_epochs <= 0:
        raise ValueError("Recording is shorter than one full epoch.")

    trimmed = signal[: n_epochs * samples_per_epoch]

    return trimmed.reshape(n_epochs, samples_per_epoch)


def _relative_bandpowers(epoch_data_uv: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """
    Compute relative EEG bandpower for all epochs of one EEG channel.

    Returns one vector per band.
    """

    frequencies, psd = welch(
        epoch_data_uv,
        fs=sfreq,
        axis=1,
        nperseg=min(epoch_data_uv.shape[1], int(4 * sfreq)),
    )

    total_mask = (frequencies >= 0.5) & (frequencies <= 30.0)
    total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=1)

    total_power = np.where(total_power <= 0, np.nan, total_power)

    outputs: dict[str, np.ndarray] = {}

    for band_name, (low, high) in BANDS.items():
        band_mask = (frequencies >= low) & (frequencies < high)
        band_power = np.trapezoid(psd[:, band_mask], frequencies[band_mask], axis=1)
        outputs[band_name] = band_power / total_power

    return outputs


def _eog_features(eog_epochs_uv: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """
    Compute locked EOG features used by the baseline v1 feature schema.
    """

    robust_range = (
        np.percentile(eog_epochs_uv, 95, axis=1)
        - np.percentile(eog_epochs_uv, 5, axis=1)
    )

    diff = np.diff(eog_epochs_uv, axis=1)
    diff_rms = np.sqrt(np.mean(diff ** 2, axis=1))

    frequencies, psd = welch(
        eog_epochs_uv,
        fs=sfreq,
        axis=1,
        nperseg=min(eog_epochs_uv.shape[1], int(4 * sfreq)),
    )

    low_mask = (frequencies >= 0.5) & (frequencies <= 4.0)
    total_mask = (frequencies >= 0.5) & (frequencies <= 30.0)

    low_power = np.trapezoid(psd[:, low_mask], frequencies[low_mask], axis=1)
    total_power = np.trapezoid(psd[:, total_mask], frequencies[total_mask], axis=1)
    total_power = np.where(total_power <= 0, np.nan, total_power)

    relative_low_power = low_power / total_power

    return {
        "EOG_Robust_Range": robust_range,
        "EOG_Diff_RMS": diff_rms,
        "EOG_Relative_Low_Power": relative_low_power,
    }


def _labels_from_annotations(
    annotations: mne.Annotations | None,
    n_epochs: int,
    epoch_length_sec: int,
) -> list[str] | None:
    """
    Convert hypnogram annotations to one label per epoch using epoch midpoint.

    Returns None if no annotations were provided.
    Unknown/movement annotations become None and are later filtered out.
    """

    if annotations is None:
        return None

    labels: list[str | None] = []

    onsets = np.asarray(annotations.onset, dtype=float)
    durations = np.asarray(annotations.duration, dtype=float)
    descriptions = list(annotations.description)

    for epoch_index in range(n_epochs):
        midpoint = epoch_index * epoch_length_sec + epoch_length_sec / 2

        label = None

        for onset, duration, description in zip(onsets, durations, descriptions):
            if onset <= midpoint < onset + duration:
                label = STAGE_DESCRIPTION_MAP.get(str(description), None)
                break

        labels.append(label)

    if all(label is None for label in labels):
        return None

    # Keep type stable for caller. Unknown labels stay as None for filtering.
    return labels  # type: ignore[return-value]


def extract_baseline_features_from_raw(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations | None = None,
    *,
    subject: str | int = "uploaded",
    recording: str | int = 0,
    epoch_length_sec: int = 30,
) -> tuple[pd.DataFrame, list[str] | None]:
    """
    Extract the 13 locked Baseline v1 features from a loaded PSG Raw object.

    This function does not train or evaluate a model.
    It only reproduces the feature table needed for temporal v2 inference.
    """

    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(epoch_length_sec * sfreq))

    if samples_per_epoch <= 0:
        raise ValueError(f"Invalid samples_per_epoch: {samples_per_epoch}")

    resolved_channels = {
        "Fpz-Cz": _resolve_channel(raw, "Fpz-Cz"),
        "Pz-Oz": _resolve_channel(raw, "Pz-Oz"),
        "horizontal": _resolve_channel(raw, "horizontal"),
    }

    # MNE returns EEG/EOG in SI units. Convert to microvolts because the
    # original notebook feature work used µV-scale values.
    channel_data_uv = {
        canonical: raw.get_data(picks=[actual])[0] * 1e6
        for canonical, actual in resolved_channels.items()
    }

    epoch_arrays = {
        channel: _epoch_signal(signal, samples_per_epoch)
        for channel, signal in channel_data_uv.items()
    }

    n_epochs = min(len(arr) for arr in epoch_arrays.values())

    epoch_arrays = {
        channel: arr[:n_epochs]
        for channel, arr in epoch_arrays.items()
    }

    labels = _labels_from_annotations(
        annotations=annotations,
        n_epochs=n_epochs,
        epoch_length_sec=epoch_length_sec,
    )

    rows: dict[str, Any] = {
        "Subject": [subject] * n_epochs,
        "Recording": [recording] * n_epochs,
        "Epoch": list(range(n_epochs)),
    }

    # Internal placeholder only. The model never uses Stage as a feature.
    if labels is None:
        rows["Stage"] = ["Wake"] * n_epochs
    else:
        rows["Stage"] = [
            label if label is not None else "Wake"
            for label in labels
        ]

    for eeg_channel in EEG_CHANNELS:
        bandpowers = _relative_bandpowers(epoch_arrays[eeg_channel], sfreq)

        for band_name in ("Delta", "Theta", "Alpha", "Sigma", "Beta"):
            rows[f"{eeg_channel}_{band_name}"] = bandpowers[band_name]

    rows.update(_eog_features(epoch_arrays[EOG_CHANNEL], sfreq))

    baseline_df = pd.DataFrame(rows)

    expected_columns = ["Subject", "Recording", "Epoch", "Stage"] + list(
        BASELINE_FEATURE_COLUMNS
    )

    baseline_df = baseline_df.loc[:, expected_columns]

    if baseline_df[list(BASELINE_FEATURE_COLUMNS)].isna().any().any():
        nan_cols = (
            baseline_df[list(BASELINE_FEATURE_COLUMNS)]
            .columns[baseline_df[list(BASELINE_FEATURE_COLUMNS)].isna().any()]
            .tolist()
        )
        raise ValueError(f"NaN values found in extracted features: {nan_cols}")

    return baseline_df, labels


def extract_temporal_features_from_raw(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations | None = None,
    *,
    subject: str | int = "uploaded",
    recording: str | int = 0,
    epoch_length_sec: int = 30,
) -> FeatureExtractionResult:
    """
    Extract baseline v1 features and expand them to temporal v2 features.
    """

    baseline_df, labels = extract_baseline_features_from_raw(
        raw=raw,
        annotations=annotations,
        subject=subject,
        recording=recording,
        epoch_length_sec=epoch_length_sec,
    )

    temporal_df = build_temporal_feature_dataframe(baseline_df)

    # If labels were not provided, do not leak the internal Stage placeholder.
    clean_labels = None if annotations is None else labels

    return FeatureExtractionResult(
        baseline_df=baseline_df,
        temporal_df=temporal_df,
        labels=clean_labels,
    )