from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mne

from core.types import ValidationResult


REQUIRED_CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
]


CHANNEL_ALIASES = {
    "EEG Fpz-Cz": [
        "EEG Fpz-Cz",
        "Fpz-Cz",
        "EEG FPZ-CZ",
        "FPZ-CZ",
    ],
    "EEG Pz-Oz": [
        "EEG Pz-Oz",
        "Pz-Oz",
        "EEG PZ-OZ",
        "PZ-OZ",
    ],
    "EOG horizontal": [
        "EOG horizontal",
        "EOG Horizontal",
        "horizontal",
        "EOG",
        "EOG horiz",
    ],
}


def _is_uploaded_file(file_or_path: Any) -> bool:
    """
    Detect Streamlit uploaded file-like objects.
    """

    return hasattr(file_or_path, "getvalue") or hasattr(file_or_path, "read")


def _write_uploaded_file_to_temp_edf(uploaded_file: Any) -> str:
    """
    Write a Streamlit uploaded EDF file to a temporary EDF file.

    MNE is more stable when reading EDF from a filesystem path.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        if hasattr(uploaded_file, "getvalue"):
            tmp.write(uploaded_file.getvalue())
        else:
            tmp.write(uploaded_file.read())

        return tmp.name


def load_psg_edf(file_or_path: str | Path | Any) -> mne.io.BaseRaw:
    """
    Load PSG EDF recording.

    This function only loads the EDF file.
    It does not run prediction, metrics, or tuning.
    """

    if _is_uploaded_file(file_or_path):
        edf_path = _write_uploaded_file_to_temp_edf(file_or_path)
    else:
        edf_path = str(file_or_path)

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        infer_types=True,
        verbose=False,
    )

    return raw


def load_hypnogram_edf(file_or_path: str | Path | Any | None) -> mne.Annotations | None:
    """
    Load hypnogram EDF annotations.

    Returns None only when no hypnogram file is provided.
    Invalid EDF annotation files should raise an error.
    """

    if file_or_path is None:
        return None

    if _is_uploaded_file(file_or_path):
        edf_path = _write_uploaded_file_to_temp_edf(file_or_path)
    else:
        edf_path = str(file_or_path)

    return mne.read_annotations(edf_path, verbose=False)


def _normalize_channel_name(name: str) -> str:
    """
    Normalize channel names for simple fuzzy matching.
    """

    return (
        name.lower()
        .replace("eeg", "")
        .replace("eog", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _find_best_channel_match(
    required_channel: str,
    available_channels: list[str],
) -> str | None:
    """
    Find matching channel name using exact aliases and normalized names.
    """

    aliases = CHANNEL_ALIASES.get(required_channel, [required_channel])

    for alias in aliases:
        if alias in available_channels:
            return alias

    normalized_available = {
        _normalize_channel_name(channel): channel
        for channel in available_channels
    }

    for alias in aliases:
        normalized_alias = _normalize_channel_name(alias)

        if normalized_alias in normalized_available:
            return normalized_available[normalized_alias]

    return None


def validate_channels(
    raw: mne.io.BaseRaw,
    required_channels: list[str] | None = None,
) -> ValidationResult:
    """
    Validate whether the PSG EDF contains the channels required by HypnoTrust.

    The app receives a ValidationResult instead of crashing.
    """

    if required_channels is None:
        required_channels = REQUIRED_CHANNELS

    available_channels = list(raw.ch_names)
    missing_required_channels: list[str] = []
    suggested_mapping: dict[str, str] = {}

    for required_channel in required_channels:
        match = _find_best_channel_match(required_channel, available_channels)

        if match is None:
            missing_required_channels.append(required_channel)
        else:
            suggested_mapping[required_channel] = match

    ok = len(missing_required_channels) == 0

    if ok:
        message = "All required PSG channels were found."
    else:
        message = (
            "Some required PSG channels are missing. "
            "Manual channel mapping may be required before running the pipeline."
        )

    return ValidationResult(
        ok=ok,
        missing_required_channels=missing_required_channels,
        available_channels=available_channels,
        suggested_mapping=suggested_mapping,
        message=message,
    )