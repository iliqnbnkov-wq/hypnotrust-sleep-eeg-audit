import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from hypnotrust_e1 import (
    AnnotationIndex, analyze_event_array, bandpower_uv2,
    event_overlap_flags, load_e0, quality_flags, window_centers,
)


class E1Tests(unittest.TestCase):
    def test_power_peak_in_band_and_outside(self):
        fs = 100
        t = np.arange(10 * fs) / fs
        slow = 20 * np.sin(2 * np.pi * 1 * t)
        fast = 20 * np.sin(2 * np.pi * 15 * t)
        pslow, _ = bandpower_uv2(slow, fs)
        pfast, _ = bandpower_uv2(fast, fs)
        self.assertGreater(pslow, pfast * 1000)

    def test_doubling_amplitude_adds_6_db(self):
        t = np.arange(1000) / 100
        x = 25 * np.sin(2 * np.pi * 1.25 * t)
        _, a = bandpower_uv2(x, 100)
        _, b = bandpower_uv2(2 * x, 100)
        self.assertAlmostEqual(b - a, 20 * np.log10(2), places=5)

    def test_window_schedule_and_crossing_boundary(self):
        self.assertEqual(len(window_centers()), 23)
        self.assertEqual(window_centers()[0], -55)
        self.assertEqual(window_centers()[-1], 55)
        t = np.arange(12000) / 100
        eeg = (15 * np.sin(2 * np.pi * t)).reshape(1, -1)
        rows = analyze_event_array(eeg, 100, 6000, channels=("EEG Fpz-Cz",))
        self.assertEqual(len(rows), 23)
        self.assertEqual([r for r in rows if r['stage_relative'] == 'mixed_at_boundary'][0]['relative_center_s'], 0)

    def test_symmetric_synthetic_signal_does_not_create_large_artificial_asymmetry(self):
        fs = 100
        t = np.arange(12000) / fs - 60
        envelope = 10 + 8 * (1 - np.exp(-np.abs(t) / 15))
        x = (envelope * np.cos(2 * np.pi * 1 * t)).reshape(1, -1)
        rows = analyze_event_array(x, fs, 6000, channels=("EEG Fpz-Cz",))
        values = {r['relative_center_s']: r['power_db_uv2'] for r in rows}
        self.assertLess(max(abs(values[t] - values[-t]) for t in values), 0.03)

    def test_annotation_grid_and_gaps(self):
        ix = AnnotationIndex([0, 90], [60, 90], ["Sleep stage 2", "Sleep stage 3"])
        self.assertEqual(ix.description_at(75), None)
        self.assertEqual(ix.description_at(95), "Sleep stage 3")
        self.assertFalse(ix.check_event({"boundary_seconds": 90,
                                         "kind": "N2->N3", "raw_n3_at_boundary": "S3"})[0])

    def test_annotation_match_for_three_epochs(self):
        ix = AnnotationIndex([0, 90], [90, 90], ["Sleep stage 2", "Sleep stage 3"])
        ev = {"boundary_seconds": 90, "kind": "N2->N3", "raw_n3_at_boundary": "S3"}
        self.assertEqual(ix.check_event(ev), (True, 'ok'))
        self.assertEqual(ix.check_event({**ev, "raw_n3_at_boundary": "S4"})[1], 'raw_n3_mismatch')

    def test_overlap_greedy_is_label_independent(self):
        e = [{"boundary_seconds": s} for s in (90, 180, 360)]
        overlap, selected = event_overlap_flags(e)
        self.assertEqual(overlap, [True, True, False])
        self.assertEqual(selected, [True, False, True])

    def test_artifact_qc(self):
        self.assertEqual(quality_flags(np.zeros(1000)), 'near_flatline')
        self.assertEqual(quality_flags(np.r_[np.zeros(999), 1000]), 'amplitude_over_limit')
        self.assertEqual(quality_flags(np.r_[np.ones(999), np.nan]), 'nonfinite')

    def test_real_e0_reference_if_present(self):
        # This test uses true E0 annotations, NOT real EEG/PSG signals.
        p = Path(__file__).parent / 'e0_transitions_reference.csv'
        if not p.exists():
            self.skipTest('optional E0 CSV not packaged')
        df = load_e0(p)
        self.assertEqual(len(df), 90)
        self.assertEqual(df.kind.value_counts().to_dict(), {'N3->N2': 47, 'N2->N3': 43})
        self.assertEqual(df.subject.nunique(), 9)
        overlap_events = 0
        for file, g in df.groupby('file'):
            flags, _ = event_overlap_flags(g.to_dict('records'))
            overlap_events += sum(flags)
        self.assertEqual(overlap_events, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class E1IntegrationTests(unittest.TestCase):
    def test_end_to_end_pipeline_mocked_edf_io(self):
        """All output paths and validation, with synthetic signal and fake EDF readers.

        Deliberately does NOT pretend that synthetic data are actual Sleep-EDF PSG.
        """
        import tempfile
        from unittest.mock import patch
        from types import SimpleNamespace
        import hypnotrust_e1

        fs = 100
        t = np.arange(300 * fs) / fs
        x = 20 * np.sin(2 * np.pi * 1.1 * t)

        class FakeRaw:
            def __init__(self):
                self.info = {"sfreq": float(fs), "meas_date": None}
                self.ch_names = list(hypnotrust_e1.CHANNELS)
                self.n_times = len(x)
            def get_data(self, picks=None, start=0, stop=None, units=None):
                assert units == 'uV'
                assert picks == list(hypnotrust_e1.CHANNELS)
                return np.stack([x[start:stop], x[start:stop] * 0.8])
            def close(self):
                pass

        a = SimpleNamespace(onset=[0, 90, 180, 270],
                            duration=[90, 90, 90, 60],
                            description=['Sleep stage 2', 'Sleep stage 3',
                                         'Sleep stage 2', 'Sleep stage ?'],
                            orig_time=None)
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            hyp = folder / 'SC4001EC-Hypnogram.edf'
            psg = folder / 'SC4001E0-PSG.edf'
            hyp.touch()
            psg.touch()
            e0 = folder / 'transitions.csv'
            pd.DataFrame([{'file':hyp.name, 'subject':'SC400', 'night':'1',
                          'kind':'N2->N3', 'boundary_seconds':90,
                          'raw_n3_at_boundary':'S3',
                          'len_before':3, 'len_after':3, 'ok_k3':1}]).to_csv(e0, index=False)
            out = folder / 'e1out'
            with patch('mne.read_annotations', return_value=a), \
                 patch('mne.io.read_raw_edf', return_value=FakeRaw()):
                result = hypnotrust_e1.run(e0, folder, out)
            self.assertEqual(result['complete_events'], 1)
            df = pd.read_csv(out / 'e1_spectral_windows.csv')
            self.assertEqual(len(df), 46)
            self.assertEqual(set(df.qc), {'ok'})
            self.assertTrue((out / 'e1_event_qc.csv').exists())
            self.assertTrue((out / 'e1_summary.json').exists())
            self.assertTrue((out / 'e1_Fpz_Cz.png').exists())
