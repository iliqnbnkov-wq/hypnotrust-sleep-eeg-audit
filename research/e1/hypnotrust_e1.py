#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HypnoTrust E1: exploratory slow-wave EEG PSD around N2/N3 label boundaries.

No changes to HypnoTrust models. Development SC400..SC408 only.
Requires numpy, scipy, pandas, mne; matplotlib optional for figures.

Example:
 py hypnotrust_e1.py --transitions "C:/.../results/pilot_e0/transitions.csv" \
   --psg-dir "C:/.../test_files/sleep_edf_real/physionet-sleep-data" \
   --out "C:/.../results/pilot_e1"
"""
from __future__ import annotations

import argparse
import json
import math
import re
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
EPOCH = 30.0
BAND = (0.5, 2.0)
HALF_WINDOW_S = 60.0
PSD_WINDOW_S = 10.0
PSD_STEP_S = 5.0
WELCH_SEGMENT_S = 4.0
WELCH_OVERLAP = 0.5
# These are exploratory signal-quality flags, not validated clinical thresholds.
PEAK_TO_PEAK_LIMIT_UV = 500.0
FLATLINE_STD_LIMIT_UV = 0.5
DEV_HYP_RE = re.compile(r"^(SC40[0-8][12])E[A-Za-z0-9]-Hypnogram\.edf$", re.I)
VALID_STAGE = {
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
}


def validate_config():
    assert PSD_WINDOW_S > WELCH_SEGMENT_S > 0
    assert 0 < PSD_STEP_S <= PSD_WINDOW_S
    assert 0 <= WELCH_OVERLAP < 1
    assert HALF_WINDOW_S >= PSD_WINDOW_S / 2
    assert (2 * HALF_WINDOW_S - PSD_WINDOW_S) / PSD_STEP_S % 1 == 0


def window_centers():
    validate_config()
    return np.arange(-HALF_WINDOW_S + PSD_WINDOW_S / 2,
                     HALF_WINDOW_S - PSD_WINDOW_S / 2 + 0.0001,
                     PSD_STEP_S, dtype=float)


def bandpower_uv2(data_uv: np.ndarray, sfreq: float) -> tuple[float, float]:
    """Welch PSD integrated over 0.5..2Hz; output (uV^2, dB re 1uV^2)."""
    data_uv = np.asarray(data_uv, dtype=float)
    if data_uv.ndim != 1 or not np.isfinite(data_uv).all():
        raise ValueError("expected finite 1D EEG window")
    if len(data_uv) < int(round(PSD_WINDOW_S * sfreq)):
        raise ValueError("too few samples")
    seg = int(round(WELCH_SEGMENT_S * sfreq))
    if seg < 8 or sfreq / seg > BAND[1] - BAND[0]:
        raise ValueError("insufficient spectral resolution")
    f, psd = welch(data_uv, fs=sfreq, window="hann", nperseg=seg,
                   noverlap=int(round(seg * WELCH_OVERLAP)),
                   detrend="constant", scaling="density")
    mask = (f >= BAND[0] - 1e-10) & (f <= BAND[1] + 1e-10)
    if mask.sum() < 2:
        raise ValueError("PSD band contains fewer than two bins")
    power = float(np.trapezoid(psd[mask], f[mask]))
    db = 10.0 * math.log10(power) if power > 0 else float("nan")
    return power, db


def quality_flags(data_uv: np.ndarray) -> str:
    """Conservative exploratory per-window QC, preserving rejected rows."""
    if not np.isfinite(data_uv).all():
        return "nonfinite"
    if np.ptp(data_uv) > PEAK_TO_PEAK_LIMIT_UV:
        return "amplitude_over_limit"
    if np.std(data_uv) < FLATLINE_STD_LIMIT_UV:
        return "near_flatline"
    return "ok"


def analyze_event_array(data_uv: np.ndarray, sfreq: float,
                        boundary_sample: int, *,
                        channels: tuple[str, ...] = CHANNELS):
    """Pure, testable signal path; data [channels, samples], boundary in samples.

    10 s local windows, centers -55..55 s; Welch uses 4 s segments with 50%
    segment overlap. A center-zero window spans the annotation boundary.
    """
    if data_uv.ndim != 2 or data_uv.shape[0] != len(channels):
        raise ValueError("expected [channels, samples]")
    rows = []
    nwin = int(round(PSD_WINDOW_S * sfreq))
    if abs(nwin - PSD_WINDOW_S * sfreq) > 1e-5:
        raise ValueError("non-integer PSD window sample count")
    for j, name in enumerate(channels):
        for center in window_centers():
            start = int(round(boundary_sample + (center - PSD_WINDOW_S / 2) * sfreq))
            stop = start + nwin
            rec = {"channel": name, "relative_center_s": float(center),
                   "stage_relative": "N2_or_N3_before" if center < 0 else
                   ("mixed_at_boundary" if center == 0 else "N2_or_N3_after")}
            if start < 0 or stop > data_uv.shape[1]:
                rec.update(qc="outside_recording", power_uv2=None, power_db_uv2=None)
                rows.append(rec)
                continue
            signal = data_uv[j, start:stop]
            qc = quality_flags(signal)
            if qc != "ok":
                rec.update(qc=qc, power_uv2=None, power_db_uv2=None)
            else:
                power, db = bandpower_uv2(signal, sfreq)
                rec.update(qc="ok" if math.isfinite(db) else "zero_band_power",
                           power_uv2=power if math.isfinite(db) else None,
                           power_db_uv2=db if math.isfinite(db) else None)
            rows.append(rec)
    return rows


def event_overlap_flags(events: list[dict], half_window=HALF_WINDOW_S):
    """Mark any overlapping pair and deterministic chronological control subset.

    Selection uses times only (no signal-derived value), separately per record.
    """
    ordered = sorted(range(len(events)), key=lambda i: (events[i]["boundary_seconds"], i))
    overlaps = [False] * len(events)
    selected = [False] * len(events)
    prev_end = -float("inf")
    for i in ordered:
        start, stop = events[i]["boundary_seconds"] - half_window, events[i]["boundary_seconds"] + half_window
        if start >= prev_end:
            selected[i] = True
            prev_end = stop
    for pos, i in enumerate(ordered):
        a = events[i]["boundary_seconds"]
        for j in ordered[pos + 1:]:
            b = events[j]["boundary_seconds"]
            if b - a >= 2 * half_window:
                break
            overlaps[i] = True
            overlaps[j] = True
    return overlaps, selected


class AnnotationIndex:
    """Read-only interval index, no interpolation between EDF annotations."""
    def __init__(self, onsets, durations, descriptions):
        entries = sorted((float(on), float(on) + float(dur), str(desc))
                         for on, dur, desc in zip(onsets, durations, descriptions))
        if any((not math.isfinite(a) or not math.isfinite(b) or b <= a) for a, b, _ in entries):
            raise ValueError("invalid annotation intervals")
        if any(entries[i][1] > entries[i + 1][0] + 0.01 for i in range(len(entries) - 1)):
            raise ValueError("overlapping annotation intervals")
        self.entries = entries
        self.starts = [e[0] for e in entries]

    def description_at(self, sec: float):
        k = bisect_right(self.starts, float(sec)) - 1
        if k < 0 or sec >= self.entries[k][1]:
            return None
        return self.entries[k][2]

    def check_event(self, event: dict, k=3) -> tuple[bool, str]:
        """Verify 3 original 30s bins on each side and boundary R&K stage."""
        b = float(event["boundary_seconds"])
        if abs(b / EPOCH - round(b / EPOCH)) > 1e-6:
            return False, "off_grid_boundary"
        left, right = event["kind"].split("->")
        if (left, right) not in (("N2", "N3"), ("N3", "N2")):
            return False, "invalid_transition_kind"
        for i in range(k):
            for side, expected in ((-1, left), (+1, right)):
                sec = b + side * (i + 0.5) * EPOCH
                if VALID_STAGE.get(self.description_at(sec)) != expected:
                    return False, "annotation_stage_mismatch"
        n3sec = b + 15 if right == "N3" else b - 15
        raw_n3 = self.description_at(n3sec)
        expected_raw = "Sleep stage 3" if event["raw_n3_at_boundary"] == "S3" else "Sleep stage 4"
        if raw_n3 != expected_raw:
            return False, "raw_n3_mismatch"
        return True, "ok"

    def scored_outside(self, duration_s):
        return [(a, b, desc) for a, b, desc in self.entries
                if b > duration_s + 0.01 and desc not in ("Sleep stage ?",)]


def load_e0(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"subject": str, "night": str, "file": str})
    required = {"file", "subject", "night", "kind", "boundary_seconds",
                "raw_n3_at_boundary", "len_before", "len_after", "ok_k3"}
    if not required.issubset(df.columns):
        raise ValueError(f"E0 CSV missing: {sorted(required - set(df.columns))}")
    if not df["file"].map(lambda f: bool(DEV_HYP_RE.fullmatch(f))).all():
        raise ValueError("E0 CSV contains unapproved subjects or wrong file names")
    for record in df.to_dict("records"):
        m = DEV_HYP_RE.fullmatch(record["file"])
        if record["subject"] != m.group(1)[:5] or str(record["night"]) != m.group(1)[5]:
            raise ValueError("subject/night metadata disagree with EDF filename")
        if record["kind"] not in ("N2->N3", "N3->N2"):
            raise ValueError("invalid E0 kind")
        if int(record["ok_k3"]) and min(int(record["len_before"]), int(record["len_after"])) < 3:
            raise ValueError("inconsistent E0 k=3 flag")
    return df[df.ok_k3 == 1].copy()


def identify_psg(hyp_file: str, psgdir: Path):
    m = DEV_HYP_RE.fullmatch(hyp_file)
    if not m:
        raise ValueError(f"not an approved SC development hypnogram: {hyp_file}")
    candidates = sorted(psgdir.rglob(m.group(1) + "*-PSG.edf"))
    if len(candidates) == 0:
        return None
    if len(candidates) != 1:
        raise ValueError(f"ambiguous PSG pair for {hyp_file}: {candidates}")
    return candidates[0]


def run(transitions_csv: Path, psgdir: Path, out: Path):
    import mne
    if not transitions_csv.is_file():
        raise FileNotFoundError(transitions_csv)
    if not psgdir.is_dir():
        raise NotADirectoryError(psgdir)
    e0 = load_e0(transitions_csv)
    out.mkdir(parents=True, exist_ok=True)
    all_windows, all_events, missing, record_qc = [], [], [], []
    for hypname, group in e0.groupby("file", sort=True):
        psg = identify_psg(hypname, psgdir)
        hyp_path = psgdir / hypname
        if psg is None or not hyp_path.is_file():
            missing.append({"hypnogram": hypname, "psg": str(psg) if psg else "missing", "hypnogram_exists": hyp_path.is_file()})
            continue
        ann = mne.read_annotations(str(hyp_path))
        index = AnnotationIndex(ann.onset, ann.duration, ann.description)
        raw = mne.io.read_raw_edf(str(psg), include=list(CHANNELS), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if not all(ch in raw.ch_names for ch in CHANNELS):
            raise ValueError(f"missing one of required EEG channels in {psg}")
        duration = raw.n_times / sfreq
        if index.entries and abs(index.entries[0][0]) > 0.01:
            raise ValueError(f"unexpected annotation start not at PSG time zero: {hypname}")
        if ann.orig_time is not None and raw.info["meas_date"] is not None:
            delta = abs((ann.orig_time - raw.info["meas_date"]).total_seconds())
            if delta > 0.01:
                raise ValueError(f"annotation and PSG absolute start differ by {delta}s")
        outside_scored = index.scored_outside(duration)
        if outside_scored:
            raise ValueError(f"scored annotation outside PSG duration: {hypname}: {outside_scored[:2]}")
        record_qc.append({"hypnogram": hypname, "psg": psg.name,
                          "sample_rate_hz": sfreq, "psg_duration_s": duration,
                          "annotation_end_s": max((e[1] for e in index.entries), default=None),
                          "annotation_orig_time": str(ann.orig_time),
                          "timebase_note": "relative EDF onset; absolute synchronization not provable" if ann.orig_time is None else "absolute origin verified"})
        events = group.sort_values("boundary_seconds").to_dict("records")
        overlaps, nonoverlap = event_overlap_flags(events)
        half_samples = int(round(HALF_WINDOW_S * sfreq))
        for i, ev in enumerate(events):
            b = float(ev["boundary_seconds"])
            eid = f"{psg.stem}:{int(b)}"
            rec = {"event_id": eid, "hypnogram": hypname, "psg": psg.name,
                   "subject": ev["subject"], "night": ev["night"],
                   "kind": ev["kind"], "boundary_seconds": b,
                   "overlap_any": bool(overlaps[i]),
                   "nonoverlap_control": bool(nonoverlap[i])}
            good, reason = index.check_event(ev, k=3)
            bsample = int(round(b * sfreq))
            if not good:
                all_events.append({**rec, "status": "invalid_annotations", "reason": reason})
                continue
            if bsample - half_samples < 0 or bsample + half_samples > raw.n_times:
                all_events.append({**rec, "status": "outside_psg", "reason": "insufficient 60s signal on both sides"})
                continue
            # Use original data on a 120s segment. units='uV' is valid for EEG channels.
            segment = raw.get_data(picks=list(CHANNELS), start=bsample-half_samples,
                                   stop=bsample+half_samples, units="uV")
            windows = analyze_event_array(segment, sfreq, half_samples)
            expected_n = len(window_centers())
            for w in windows:
                all_windows.append({**rec, **w})
            counts = Counter(w["channel"] for w in windows if w["qc"] == "ok")
            event_clean = all(counts[ch] == expected_n for ch in CHANNELS)
            all_events.append({**rec, "status": "complete" if event_clean else "partial_qc",
                               "reason": "ok" if event_clean else "one or more windows excluded",
                               **{f"clean_windows_{ch.split()[-1]}": counts[ch] for ch in CHANNELS}})
        raw.close()
    dfw = pd.DataFrame(all_windows)
    dfe = pd.DataFrame(all_events)
    dfw.to_csv(out / "e1_spectral_windows.csv", index=False)
    dfe.to_csv(out / "e1_event_qc.csv", index=False)
    pd.DataFrame(record_qc).to_csv(out / "e1_record_validation.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "e1_missing_pairs.csv", index=False)
    summary = {
        "name": "HypnoTrust E1 exploratory spectral pilot",
        "e0_eligible_k3": len(e0),
        "matched_psg": len(record_qc),
        "missing_records": len(missing),
        "processed_events": len(dfe),
        "complete_events": int((dfe.status == "complete").sum()) if len(dfe) else 0,
        "overlap_flagged_events": int(dfe.overlap_any.sum()) if len(dfe) else 0,
        "nonoverlap_control_events": int(dfe.nonoverlap_control.sum()) if len(dfe) else 0,
        "analysis_parameters": {
            "channels": CHANNELS, "band_hz": BAND, "psd_window_s": PSD_WINDOW_S,
            "window_step_s": PSD_STEP_S, "welch_segment_s": WELCH_SEGMENT_S,
            "welch_overlap": WELCH_OVERLAP, "event_window_s": [-HALF_WINDOW_S, HALF_WINDOW_S],
            "main_k": 3, "units": "uV^2; 10log10(uV^2 / 1uV^2)",
            "qc_peak_to_peak_uv": PEAK_TO_PEAK_LIMIT_UV,
            "qc_min_std_uv": FLATLINE_STD_LIMIT_UV,
        },
        "limitations": [
            "Exploratory k=3 was selected after reviewing E0 annotation counts.",
            "Annotations are R&K scored; N3 combines S3/S4.",
            "Stage boundaries are 30s label boundaries, not precise physiological transitions.",
            "EDF hypnogram orig_time=None means absolute time origin cannot independently be verified.",
            "Overlapping event windows and repeated events per participant are dependent.",
            "QC thresholds are exploratory and should be sensitivity-tested.",
            "No population inference is justified with a single participant.",
        ],
    }
    (out / "e1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(dfw) and len(dfe):
        make_plots(dfw, dfe, out)
    print(json.dumps({k: summary[k] for k in ("e0_eligible_k3", "matched_psg", "missing_records", "processed_events", "complete_events", "overlap_flagged_events", "nonoverlap_control_events")}, indent=2))
    print(f"Outputs: {out}")
    return summary


def make_plots(windows, events, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    clean_ids = set(events.loc[events.status == "complete", "event_id"])
    data = windows.loc[windows.event_id.isin(clean_ids) & (windows.qc == "ok")].copy()
    if data.empty:
        return
    # Descriptive only: individual profiles plus mean; no false precision/CI.
    for channel in CHANNELS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for kind in ("N2->N3", "N3->N2"):
            dd = data[(data.channel == channel) & (data.kind == kind)]
            if dd.empty:
                continue
            for _, event in dd.groupby("event_id"):
                event = event.sort_values("relative_center_s")
                ax.plot(event.relative_center_s, event.power_db_uv2, alpha=0.14, lw=0.9)
            mean = dd.groupby("relative_center_s").power_db_uv2.mean()
            ax.plot(mean.index, mean.values, lw=2.5,
                    label=f"{kind} (n={dd.event_id.nunique()} events)")
        ax.axvline(0, color="black", ls="--", lw=1, label="annotated boundary")
        ax.set(xlabel="Seconds relative to 30s label boundary",
               ylabel="0.5–2 Hz power, dB re 1 µV²",
               title=f"E1 exploratory event-aligned PSD — {channel}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / ("e1_" + channel.split()[-1].replace("-", "_") + ".png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True, help="E0 transitions.csv")
    parser.add_argument("--psg-dir", type=Path, required=True, help="folder containing PSG and hypnogram EDF pairs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.transitions, args.psg_dir, args.out)


if __name__ == "__main__":
    main()
