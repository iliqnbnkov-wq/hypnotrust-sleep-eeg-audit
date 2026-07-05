# HypnoTrust: Uncertainty-Aware Review Prioritization for Sleep EEG Staging

HypnoTrust is **not** presented as a new sleep-stage classifier.

It is a proof-of-concept **trust and audit workflow** for automatic sleep staging. The project focuses on uncertainty-aware review prioritization, calibration checks, locked evaluation, external holdout testing, failure-mode analysis and a Streamlit dashboard for inspecting model behavior.

> Educational / research proof-of-concept only. Not intended for clinical use.

## Core idea

Most automatic sleep staging projects focus on one question:

> What did the model predict?

HypnoTrust focuses on the next layer:

> When should we trust the prediction less, and which epochs should be prioritized for human review?

The dashboard does not train or tune models during the demo. It reads frozen prediction outputs from the controlled V5d evaluation workflow.

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

- Historical locked subject evaluation: accuracy around 0.81-0.83 depending on the evaluation version.
- External holdout pooled evaluation: accuracy around 0.78.
- Review queues are enriched for errors compared with the full epoch set.
- Common failure modes include REM -> N2 and N3 -> N2 behavior.
- The project explicitly tracks N2-attractor behavior and subject-level variability.

These results are used for inspection and explanation, not for clinical claims.

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
```

## Documentation

- [Project positioning](docs/project_positioning.md)
- [Related work](docs/related_work.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Model card summary](docs/model_card.md)

## Quick start on Windows

After cloning the repository:

```powershell
cd hypnotrust-sleep-eeg-audit
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_app.ps1
```

The launch script automatically sets:

```powershell
$env:PYTHONPATH = "$RepoRoot;$RepoRoot\src"
```

This is required because the Streamlit app uses both the repository root and `src/` modules.

## Run the app

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
.\run_app.ps1
```

## Run the app

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_app.ps1
```

`run_app.ps1` creates/uses the local `.venv`, installs dependencies from `requirements.txt` if Streamlit is missing, sets `PYTHONPATH`, and launches the Streamlit dashboard.

## Run tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p "test*.py"
```

Expected local test status for the current project version:

```text
Ran 133 tests
OK
```

## Limitations

- The project is based on a limited research dataset and frozen derived outputs.
- The dashboard is an audit and explanation tool, not a clinical system.
- Confidence is useful for review prioritization, but it is not a guarantee of correctness.
- Subject-level variability remains important.
- Failure modes such as N2-attractor behavior require careful interpretation.

## Status

Current stable version: `v0.1.0`

Research / educational proof-of-concept.
