# Reproducibility notes

Status at audit date 2026-11-22, `main` = `fcd9a5f5bc0d87745d48c6134421849039d4f438`.

## What can be reproduced from this repository

- Unit and integration tests that use only repository files and synthetic data (run with `python -m pytest -q`).
- Reading the frozen V5d outputs and recomputing accuracy, macro-F1 and Cohen's kappa from `v5d_epoch_predictions.csv`.
- Running the dashboard on the frozen artifacts.

## What cannot be reproduced from this repository

| Item | Status |
|---|---|
| Raw Sleep-EDF PSG/hypnogram files | Not included (data policy, `.gitignore`). Tests that need SC4001 data are separated below. |
| Training of the Temporal v2 model | Not provided as a script in the repository. NOT VERIFIED. |
| Original Git commit and Python environment of V5d | Not recorded. NOT VERIFIED. |
| `v5d_subject_metrics.csv`, `v5d_selective_prediction.csv`, `v5d_per_stage_metrics.csv` | Listed in the V5d report, absent from the repository. Subject-level accuracy was recomputed from the predictions file; the 80%-coverage selective-prediction figures could only be approximated (ties/threshold rule not documented) and are NOT VERIFIED exactly. |
| Version-selection history for Subject 9 | Not in the repository. Independence of later decisions from Subject 9 cannot be assessed. |

## Tests that need external data or runtime artifacts

Tests whose names contain `sc4001` (`tests/test_sc4001_alignment.py`, `tests/test_edf_integration_sc4001.py`, `tests/test_transition_integration.py`) expect a local Sleep-EDF recording. They must not be required in CI. Check each test's skip logic before relying on a pass: a test that returns early without data is not evidence that the pipeline works.

## Open decisions for the repository owner

1. **License.** No `LICENSE` exists. Choose one deliberately (for example MIT/Apache-2.0 for code); the choice is not made in this audit.
2. **Release.** Tags `v0.1.0` and `v0.1.1` exist; no GitHub Release exists. README previously stated `v0.1.0`.
3. **Pickle artifacts.** `.joblib` files are pickles; consider publishing hashes (SHA-256) next to them. Hashes at audit time:
   - `results/calibration_v3b/calibrator_v3b.joblib`: `4ee699530afe403c12dfea723ca4edc53fe5233a65a216b814c224ae40d4f484`
   - `results/temporal_v2/sleep_stage_random_forest_temporal_v2.joblib`: `e9f22f087743cbd32b50a73d7368493c9c003fa90364d442526a7b13eae5dcb9`

## Dependency advisories (pip-audit, 2026-11-22)

`pip-audit -r requirements.txt` reported 89 advisories in 5 pinned packages: pillow 12.3.0 (42), gitpython 3.1.50 (39), urllib3 2.7.0 (3), anyio 4.14.1 (3), certifi 2026.6.17 (2). The advisory feed was read in the audit sandbox and individual advisories were not reviewed; relevance to this application (for example GitPython is a Streamlit dependency and is not used for user input) is NOT VERIFIED. Upgrading pins changes the environment and should be done deliberately and re-tested; this audit did not change `requirements.txt`. The workflow `security.yml` runs the check as non-blocking.
