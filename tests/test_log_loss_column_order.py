"""Regression test: log loss must respect probability column order."""

from __future__ import annotations

import numpy as np

from core.metrics import _log_loss_in_stage_order

STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def test_log_loss_matches_manual_formula_in_stage_order():
    y_true = np.array(STAGES * 20)
    index = np.array([STAGES.index(s) for s in y_true])
    probabilities = np.eye(5)[index] * 0.96 + 0.01

    expected = float(-np.mean(np.log(probabilities[np.arange(len(y_true)), index])))

    assert abs(_log_loss_in_stage_order(y_true, probabilities, STAGES) - expected) < 1e-9
