"""
Test confusion matrix axis labels to ensure rows=TRUE, columns=PREDICTED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_confusion_matrix_axis_labels():
    """Verify confusion matrix has correct axis labels."""
    from core.metrics import build_confusion_matrix_df

    # Test data matching SC4001 reference
    y_true = ["Wake", "Wake", "N1", "N2", "N3", "REM"]
    y_pred = ["Wake", "N1", "N1", "N2", "N3", "REM"]
    stage_names = ["Wake", "N1", "N2", "N3", "REM"]

    cm_df = build_confusion_matrix_df(y_true, y_pred, stage_names)

    print("Confusion Matrix DataFrame:")
    print(cm_df)
    print()

    # Check index (rows) - should be true_{stage}
    row_labels = list(cm_df.index)
    print(f"Row labels: {row_labels}")
    assert all(label.startswith("true_") for label in row_labels), "All row labels must start with 'true_'"

    # Check columns - should be pred_{stage}
    col_labels = list(cm_df.columns)
    print(f"Column labels: {col_labels}")
    assert all(label.startswith("pred_") for label in col_labels), "All column labels must start with 'pred_'"

    # Verify the numeric matrix is correct
    # Expected confusion matrix for this test data:
    #          pred_Wake  pred_N1  pred_N2  pred_N3  pred_REM
    # true_Wake       1        1        0        0         0
    # true_N1         0        1        0        0         0
    # true_N2         0        0        1        0         0
    # true_N3         0        0        0        1         0
    # true_REM        0        0        0        0         1

    assert cm_df.loc["true_Wake", "pred_Wake"] == 1
    assert cm_df.loc["true_Wake", "pred_N1"] == 1
    assert cm_df.loc["true_N1", "pred_N1"] == 1
    assert cm_df.loc["true_N2", "pred_N2"] == 1
    assert cm_df.loc["true_N3", "pred_N3"] == 1
    assert cm_df.loc["true_REM", "pred_REM"] == 1

    print("[PASS] Confusion matrix axis labels are correct (rows=TRUE, columns=PREDICTED)")
    return True


if __name__ == "__main__":
    success = test_confusion_matrix_axis_labels()
    sys.exit(0 if success else 1)
