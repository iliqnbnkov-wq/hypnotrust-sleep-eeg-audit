"""Regression tests for temporary-file handling around lazily read EDF files.

These tests build a tiny synthetic EDF in a temporary directory. They use no
Sleep-EDF data, no trained model and no locked subject.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
pytest.importorskip("edfio")

from core.loader import load_psg_edf, preload_eeg_channels  # noqa: E402


def _write_synthetic_edf(path: Path) -> None:
    info = mne.create_info(
        ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"], 100.0, "eeg"
    )
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(size=(3, 6000)) * 1e-5, info, verbose="ERROR")
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="ERROR")


def test_lazy_raw_breaks_after_file_deletion(tmp_path):
    """Documents the original defect: lazy Raw + deleted file -> error."""
    edf = tmp_path / "s.edf"
    _write_synthetic_edf(edf)
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
    os.remove(edf)
    with pytest.raises(FileNotFoundError):
        raw.get_data(start=0, stop=100)


def test_preload_eeg_channels_survives_file_deletion(tmp_path):
    edf = tmp_path / "s.edf"
    _write_synthetic_edf(edf)
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
    raw = preload_eeg_channels(raw)
    os.remove(edf)
    data = raw.get_data(start=0, stop=100)
    assert data.shape == (2, 100)
    assert raw.ch_names == ["EEG Fpz-Cz", "EEG Pz-Oz"]


def test_preload_keeps_all_channels_when_none_match(tmp_path):
    edf = tmp_path / "s.edf"
    _write_synthetic_edf(edf)
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
    raw = preload_eeg_channels(raw, channel_keys=("does-not-exist",))
    assert len(raw.ch_names) == 3


class _Upload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def test_load_psg_edf_removes_temp_copy_of_upload(tmp_path, monkeypatch):
    edf = tmp_path / "s.edf"
    _write_synthetic_edf(edf)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))

    raw = load_psg_edf(_Upload(edf.read_bytes()))

    assert raw.get_data().shape[0] == 3
    assert list(scratch.iterdir()) == []
