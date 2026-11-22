# HypnoTrust: Uncertainty-Aware Review Prioritization for Sleep EEG Staging

HypnoTrust is **not** presented as a new sleep-stage classifier.

It is a proof-of-concept **trust and audit workflow** for automatic sleep staging. The project focuses on uncertainty-aware review prioritization, calibration checks, locked evaluation, external holdout testing, failure-mode analysis and a Streamlit dashboard for inspecting model behavior.

> Educational / research proof-of-concept only. Not intended for clinical use.

## Core idea

Most automatic sleep staging projects focus on one question:

> What did the model predict?

HypnoTrust focuses on the next layer:

> When should we trust the prediction less, and which epochs should be prioritized for human review?

During normal operation, the dashboard does not train, calibrate, or tune models. EDF inputs can invoke `core.pipeline.run_full_pipeline` for inference with the existing locked Temporal v2 model. Separate comparison panels read frozen prediction outputs from the public V5d evaluation workflow.

## What the project demonstrates

- Automatic sleep-stage prediction inspection
- Confidence and uncertainty analysis
- Low-confidence review queue
- Review queue CSV export
- Stage distribution analysis
- Error-pair / failure-mode analysis
- Subject-level comparison dashboard
- Frozen evaluation outputs from a controlled V5d workflow
- Clear proof-of-concept / no-clinical-use framing

## Demo workflow

The Streamlit app supports the following review flow:

1. Select an evaluation role or subject.
2. Inspect overall metrics such as accuracy, confidence and review-queue behavior.
3. Compare predicted sleep stages against the reference hypnogram.
4. Inspect low-confidence epochs in the review queue.
5. Export the review queue as CSV.
6. Compare true and predicted stage distributions.
7. Inspect common error pairs such as REM -> N2 or N3 -> N2.
8. Compare subject-level behavior across the frozen evaluation set.

## Selected frozen results

The project keeps model-development and evaluation roles separated. Selected locked/frozen results include:

- Review queues are enriched for errors compared with the full epoch set.
- The V5d error-pair tables include REM -> N2 and N3 -> N2 among frequent errors.
- The project reports subject-level variability.

Role-specific figures from the public V5d report:

- **Historical locked evaluation:** SC409 (Subject 9), 1,132 epochs; V5d reconstruction accuracy 0.8127. The report compares this with the historical Temporal v2 reference accuracy 0.8313.
- **V5d external holdout:** SC410-SC419, 10,713 pooled epochs; reported pooled accuracy 0.7752, macro-F1 0.6947, and Cohen's kappa 0.6692.

These are values reported by checked-in artifacts, not results independently reproduced in this checkout session. The historical locked evaluation and external holdout are distinct; neither is clinical validation.

## What the project does not claim

- It does not claim clinical validation.
- It does not claim state-of-the-art performance.
- It does not replace sleep specialists.
- It does not claim that automatic sleep staging is new.
- It is not intended for real medical use.

## Repository structure

```text
app/
  app.py                         Streamlit dashboard
core/                             EDF-to-prediction application pipeline
configs/                          Baseline experiment configuration
results/                          Temporal v2 and calibration artifacts
research/e1/                      Separate E1 research experiment

src/hypnotrust/
  calibration.py                 Calibration and probability handling
  constants.py                   Shared project constants
  io.py                          Input/output helpers
  temporal.py                    Temporal context helpers
  uncertainty.py                 Confidence and selective prediction logic
  validation.py                  Validation and safety checks

tests/
  test_*.py                      Unit tests

docs/
  project_positioning.md         Project framing
  related_work.md                Existing work and where HypnoTrust fits
  evaluation_protocol.md         Evaluation and no-leakage framing
  model_card.md                  Intended use, risks and limitations

experiments/
  v5_expanded_cross_subject_evaluation/
    v5d_frozen_external_evaluation/
      frozen CSV/JSON/MD evaluation outputs

.github/                         Copilot instructions and repository Agent Skills
run_app.ps1                       App launcher (creates environment and installs/updates packages)
run_tests.ps1                     Test launcher (creates environment and installs/updates packages)
```

## Documentation

- [Project positioning](docs/project_positioning.md)
- [Related work](docs/related_work.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Model card summary](docs/model_card.md)
- [V5d frozen external evaluation report](experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/V5D_FROZEN_EXTERNAL_EVALUATION_REPORT.md)
- [V5d evaluation configuration](experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/v5d_eval_config.json)

## Windows setup and app

The app and test PowerShell scripts create `.venv` when needed and install/update packages from `requirements.txt`. To avoid implicit installation, provision dependencies explicitly:

```powershell
cd hypnotrust-sleep-eeg-audit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$env:PYTHONPATH = "$PWD;$PWD\src"
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py
```

Alternatively, `.\run_app.ps1` installs/updates packages automatically before starting the app.

## Test command

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"
```

Run this only in an already provisioned environment. The command above does not install dependencies; `.\run_tests.ps1` does install/update packages and should not be used when installation is not intended.

The README previously recorded `Ran 133 tests / OK`; this is a repository-reported status, not an independently reproduced result in this checkout session.

## Limitations

- The project is based on a limited research dataset and frozen derived outputs.
- The dashboard is an audit and explanation tool, not a clinical system.
- Confidence is useful for review prioritization, but it is not a guarantee of correctness.
- Subject-level variability remains important.
- N2-related error pairs require careful interpretation.

## Status

Current stable version: `v0.1.0`

Research / educational proof-of-concept.
