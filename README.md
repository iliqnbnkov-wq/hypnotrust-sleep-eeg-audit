# HypnoTrust: Uncertainty-Aware Review Prioritization for Sleep EEG Staging

HypnoTrust is not presented as a new sleep-stage classifier.

It is a proof-of-concept trust and audit workflow for automatic sleep staging. The project focuses on uncertainty-aware review prioritization, calibration checks, locked evaluation, external holdout testing, failure-mode analysis and a Streamlit dashboard for inspecting model behavior.

## What the project demonstrates

- Automatic sleep-stage prediction inspection
- Confidence and uncertainty analysis
- Low-confidence review queue
- Review queue CSV export
- Stage distribution analysis
- Error-pair / failure-mode analysis
- Subject-level comparison dashboard
- Frozen evaluation outputs from a controlled V5d workflow

## What the project does not claim

- It does not claim clinical validation.
- It does not claim state-of-the-art performance.
- It does not replace sleep specialists.
- It does not claim that automatic sleep staging is new.
- It is not intended for real medical use.

## Evaluation framing

The project uses frozen predictions and a controlled evaluation protocol. The Streamlit app reads existing prediction outputs and does not train or tune models during the demo.

## Run the app

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run .\app\app.py
```

## Run tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p "test*.py"
```


## Documentation

- [Project positioning](docs/project_positioning.md)
- [Related work](docs/related_work.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [Model card summary](docs/model_card.md)

## Status

Research / educational proof-of-concept.

