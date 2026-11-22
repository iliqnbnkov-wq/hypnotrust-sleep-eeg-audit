# HypnoTrust repository guidance

HypnoTrust is a research and educational proof of concept for uncertainty-aware review prioritization of sleep-stage predictions (Sleep-EDF). It is not clinically validated and must never be described as a medical device or as state of the art.

## Data roles (do not blur)

- Development: SC400–SC408. Historical locked subject: SC409 ("Subject 9"), which has an evaluation history and must not be called untouched. External holdout: SC410–SC419.
- Never use SC409 or SC410–SC419 for training, tuning, feature selection, calibration, threshold or alpha selection, review-policy design or model-version selection.

## Hard rules

- Do not invent metrics, test results, citations or conclusions. Every reported number needs a file, protocol and aggregation level (pooled epochs vs subject-level).
- Do not modify frozen artifacts: `results/**`, `experiments/**`, `*.joblib`, locked reports. Do not train models or regenerate outputs.
- Do not rewrite scientific conclusions automatically.
- Every code change needs a test. Do not claim a test passed unless it was run.
- Keep inference and evaluation separate; the app must not train or tune.
- Mark anything that cannot be checked as NOT VERIFIED.

## Working conventions

- Python code under `core/` and `src/hypnotrust/`; Streamlit app in `app/app.py`; tests in `tests/` (`python -m pytest -q` with `PYTHONPATH` set to the repo root and `src`).
- Tests must not need raw PSG/EDF files unless clearly marked and skipped when data is absent.
- Reports for the project owner are written in Bulgarian; code, identifiers and repository docs stay in English.
