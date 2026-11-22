# HypnoTrust: Uncertainty-Aware Review Prioritization for Sleep EEG Staging

> **Research and educational proof of concept. Not a medical device, not clinically validated, not intended for diagnosis or clinical decision-making.**

## 1. Project overview

HypnoTrust is a trust-and-audit workflow built around an automatic sleep-stage classifier. It is **not** presented as a new or state-of-the-art classifier. Its subject is the question that follows a prediction: *when should the prediction be trusted less, and which epochs should a human reviewer look at first?*

The repository contains:

- a feature-based sleep-stage model (frozen artifacts) for 30-second epochs;
- confidence, entropy, calibration and selective-prediction utilities;
- a low-confidence review queue;
- frozen evaluation outputs (V5d) and their documentation;
- a Streamlit dashboard for inspecting predictions and running the frozen pipeline on EDF files;
- unit and integration tests.

## 2. Scientific motivation

Automatic sleep staging is an established field (see [References](#15-references)). Published work already combines sequence modelling and uncertainty-based deferral of low-confidence epochs to a human expert (for example SleepTransformer). HypnoTrust does not claim novelty for staging itself. It documents a small, inspectable workflow in which subject roles are separated, calibration is fitted without the locked subject, and failure modes (for example the tendency to predict N2) are reported next to headline metrics.

## 3. What HypnoTrust contributes

- A transparent evaluation hierarchy with explicit subject roles (see [section 9](#9-evaluation-protocol)).
- Review prioritization based on prediction confidence and entropy, with selective-prediction curves.
- Out-of-fold probability calibration code that refuses Subject 9 as input (`src/hypnotrust/calibration.py`).
- Frozen, versioned outputs and tests that guard the 13-feature baseline contract.

It does **not** claim clinical validation, state-of-the-art accuracy, replacement of sleep specialists, or novelty of automatic sleep staging.

## 4. Dataset and signals

- **Data source:** Sleep-EDF Expanded (Sleep Cassette study, subjects SC400–SC419 are referenced in the V5d configuration). Raw PSG/EDF files are **not** included in this repository and are excluded by `.gitignore`.
- **Signals used:** `EEG Fpz-Cz`, `EEG Pz-Oz` and `EOG horizontal`.
- **Epochs:** 30 s. Sampling frequency in the configuration: 100 Hz (`configs/baseline_v1.yaml`).
- **Labels:** Wake, N1, N2, N3, REM.

Counting units matter and are kept separate in this repository: *participants/subjects* (SC400–SC419), *recordings* and *30-second epochs*. For example, V5d reports 8,781 development epochs, 1,132 epochs for Subject 9 (SC409) and 10,713 epochs for SC410–SC419 (`v5d_eval_config.json`). The V5d predictions file contains one recording per subject.

## 5. Processing pipeline

1. Read EDF with MNE; resolve the three required channels (`core/loader.py`).
2. Band-pass filtering (EEG 0.5–40 Hz, EOG 0.3–15 Hz) and Welch spectra (`configs/baseline_v1.yaml`).
3. **13 baseline features:** relative band powers (Delta, Theta, Alpha, Sigma, Beta) for each of the two EEG channels plus three EOG features (`src/hypnotrust/constants.py`).
4. **Temporal context:** features of epochs t−2…t+2 are concatenated, giving **65 features** (Temporal v2).
5. Random-forest probabilities → optional calibration → confidence/entropy → review queue.

## 6. Model architecture

A `RandomForestClassifier` on the 65 temporal features. The V5d configuration records 300 trees, `class_weight="balanced_subsample"` and `random_state=42`. The trained artifact is `results/temporal_v2/sleep_stage_random_forest_temporal_v2.joblib`. See the [model card](docs/model_card.md).

## 7. Uncertainty and calibration

- Confidence is the maximum class probability; entropy is computed from the probability vector (`src/hypnotrust/uncertainty.py`).
- Calibration v3b is a multiclass Platt-style calibrator fitted on **out-of-fold** probabilities of the nine development subjects (SC400–SC408; `StratifiedGroupKFold`, `src/hypnotrust/calibration.py`). The code raises an error if Subject 9 is present in fitting data. The fitted calibrator is `results/calibration_v3b/calibrator_v3b.joblib`; calibration is **off by default** in the application configuration (`apply_calibration=False`, `core/config.py`).
- Selective prediction reports accuracy on the retained epochs at requested coverages (`selective_performance_curve`).

## 8. Review prioritization

`core/review_queue.py` ranks epochs by ascending confidence and returns the lowest-confidence fraction (default 20%, `DEFAULT_REVIEW_FRACTION`). Two uncertainty policies exist in the type system: `v3` (default) and `v4_stage_aware`. A review queue is a prioritization aid; confidence is not a guarantee of correctness.

## 9. Evaluation protocol

| Role | Subjects | Permitted use |
|---|---|---|
| Development | SC400–SC408 (nine participants) | Training, out-of-fold calibration |
| Historical locked test subject | SC409 ("Subject 9") | Reported, not used for fitting; has a **history of repeated evaluation** across model versions |
| External holdout (V5d) | SC410–SC419 | Reported once under the V5d protocol |

Subject 9 must not be described as an untouched test subject for future decisions: it was evaluated repeatedly in earlier versions, so any version, threshold or policy choice made after seeing those results is not independent of it. Details: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## 10. Verified experimental results

All values below are copied from `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/` (report, `v5d_eval_config.json`, `v5d_external_summary.csv`). Accuracy, macro-F1 and Cohen's kappa for the pooled and Subject 9 rows were recomputed from `v5d_epoch_predictions.csv` during an audit on 2026-11-22 and match the report; nothing was re-run on a model.

| Evaluation | Aggregation | Epochs | Accuracy | Macro-F1 | Cohen's kappa |
|---|---|---:|---:|---:|---:|
| V5d, SC409 (historical locked subject) | single subject | 1,132 | 0.8127 | 0.7333 | 0.7120 |
| V5d, SC410–SC419 (external holdout) | pooled over epochs | 10,713 | 0.7752 | 0.6947 | 0.6692 |
| V5d, SC410–SC419 | mean ± SD over 10 subjects (SD with ddof = 1) | 928–1,535 per subject | 0.7768 ± 0.0641 | 0.6600 ± 0.0785 | 0.6688 ± 0.1001 |

Per-subject accuracy in the external holdout ranges from 0.6707 (SC417) to 0.8498 (SC415). The report itself describes a measurable cross-subject drop relative to Subject 9. Other values (for example the locked Temporal v2 reference accuracy 0.8313 and the 80%-coverage selective-prediction figures) are in the V5d report; the three CSV files listed in that report as `v5d_subject_metrics.csv`, `v5d_selective_prediction.csv` and `v5d_per_stage_metrics.csv` are **not present in this repository** (see [limitations](#14-scientific-limitations)).

## 11. Installation and usage

The pinned versions in `requirements.txt` (NumPy 2.5, pandas 3.0, scikit-learn 1.9) require a recent Python. The audit ran the tests on Python 3.13 with slightly newer patch versions than the pins; the original V5d interpreter version is not recorded.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src"    # PowerShell: $env:PYTHONPATH = "$PWD;$PWD\src"
streamlit run app/app.py
```

On Windows, `run_app.ps1` creates `.venv`, installs `requirements.txt` if needed and starts the app; `run_tests.ps1` installs packages and runs the tests. Both modify the local environment.

**Application modes** (sidebar):

- **User** – upload a feature CSV (baseline or Temporal v2 columns); predictions only, no accuracy is shown.
- **Demo** – synthetic profiles run through the real model.
- **EDF Upload** (experimental) – upload a PSG EDF (and an optional hypnogram EDF); the frozen pipeline extracts features, predicts, computes uncertainty and builds a review queue.
- **Research** – full metrics when verified labels are present.
- **Transition Explorer** – detect N2↔N3 boundaries from a hypnogram and inspect spectral power around them.

The dashboard does not train or tune models; it loads frozen artifacts.

**Tests**

```bash
export PYTHONPATH="$PWD:$PWD/src"
python -m pytest -q
```

Tests that need external data or runtime artifacts are documented in [docs/reproducibility.md](docs/reproducibility.md).

## 12. Repository structure

```text
app/                 Streamlit dashboard
core/                EDF loading, feature extraction, pipeline, review queue
src/hypnotrust/      calibration, uncertainty, validation, temporal context
configs/             baseline_v1.yaml
tests/               unit and integration tests
docs/                protocol, model card, positioning, related work, reproducibility
experiments/         V5d frozen external evaluation outputs
results/             frozen Temporal v2 / calibration v3b artifacts
research/e1/         exploratory E1 spectral pilot (separate requirements)
.github/             Copilot instructions, skills, CI workflows
```

## 13. Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md). In short: dependencies are pinned in `requirements.txt`; frozen artifacts are in the repository; the exact original V5d checkout and Python environment are **not recorded** and are therefore not claimed.

## 14. Scientific limitations

- Small development cohort (nine participants) and a single historical locked subject; between-subject variability is large (SD of per-subject accuracy ≈ 0.06 in the external holdout).
- Subject 9 has an evaluation history (section 9).
- Reported results are from one dataset and one recording per subject; no external dataset or clinical population was evaluated.
- Known failure modes: predictions drift towards N2 (for example REM→N2 and N3→N2), and confidence can be mismatched with correctness.
- Pooled (epoch-level) and subject-level metrics differ; the pooled value gives more weight to longer recordings. Confidence intervals are not reported.
- Three CSV artifacts named in the V5d report are missing from the repository; the original Git commit and environment of V5d are not recorded.
- Committed `.joblib` files are pickles: load them only from a trusted source (this repository at a verified commit).
- Not a clinical tool.

## 15. References

- Supratak A, Dong H, Wu C, Guo Y. DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG. *IEEE Trans Neural Syst Rehabil Eng.* 2017;25(11):1998–2008. [doi:10.1109/TNSRE.2017.2721116](https://doi.org/10.1109/TNSRE.2017.2721116)
- Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging. *IEEE Trans Neural Syst Rehabil Eng.* 2019;27(3):400–410. [doi:10.1109/TNSRE.2019.2896659](https://doi.org/10.1109/TNSRE.2019.2896659)
- Phan H, Mikkelsen K, Chen OY, Koch P, Mertins A, De Vos M. SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification. *IEEE Trans Biomed Eng.* 2022;69(8):2456–2467. [doi:10.1109/TBME.2022.3147187](https://doi.org/10.1109/TBME.2022.3147187)
- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Trans Biomed Eng.* 2000;47(9):1185–1194. [doi:10.1109/10.867928](https://doi.org/10.1109/10.867928) (the reference for the Sleep-EDF database)

Bibliographic details were checked against PubMed. U-PASS is discussed in [docs/related_work.md](docs/related_work.md) without a verified citation.

## 16. Citation and licensing

Citation metadata: [CITATION.cff](CITATION.cff).

**License: not yet specified.** No `LICENSE` file is present, so by default all rights are reserved and others have no permission to reuse the code. The repository owner has to choose a license (see [docs/reproducibility.md](docs/reproducibility.md) for the open decision). Sleep-EDF data are distributed by PhysioNet under their own terms; derived feature and prediction files in this repository remain subject to those terms.
