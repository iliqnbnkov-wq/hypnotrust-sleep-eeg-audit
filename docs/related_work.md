# Related Work and Project Positioning

## Why this section exists

Automatic sleep staging is an established research area. HypnoTrust does not claim that sleep-stage classification itself is new.

The goal of this project is different: to build a small, reproducible trust and audit workflow around sleep-stage predictions.

## Existing work

### DeepSleepNet

DeepSleepNet is a deep learning model for automatic sleep stage scoring based on raw single-channel EEG. It combines convolutional feature extraction with temporal transition learning.

### SeqSleepNet

SeqSleepNet frames automatic sleep staging as a sequence-to-sequence problem rather than isolated epoch classification.

### SleepTransformer

SleepTransformer combines automatic sleep staging with interpretability and uncertainty quantification. Its uncertainty score can be used to defer low-confidence epochs for human inspection.

### U-PASS

U-PASS is an uncertainty-guided deep learning pipeline for automated sleep staging and human-centered review workflows.

## Where HypnoTrust fits

HypnoTrust is not positioned as a new state-of-the-art classifier.

Instead, HypnoTrust demonstrates:

- frozen prediction inspection
- uncertainty-aware review prioritization
- confidence and calibration-oriented analysis
- locked evaluation framing
- external holdout reporting
- failure-mode analysis
- subject-level comparison
- Streamlit dashboard for explanation and audit

## Core claim

HypnoTrust does not ask only: What did the model predict?

It asks:

- When is the model uncertain?
- Which epochs should be prioritized for review?
- What types of mistakes repeat?
- Does the model show subject-specific weaknesses?
- Can the prediction workflow be audited transparently?

## What HypnoTrust does not claim

- It does not claim clinical validation.
- It does not claim state-of-the-art performance.
- It does not replace sleep specialists.
- It does not claim that automatic sleep staging is new.
- It is not intended for real medical use.

## Defense positioning

If asked whether automatic sleep staging already exists, the correct answer is:

> Yes. Automatic sleep staging and uncertainty-aware methods already exist. This project does not claim otherwise. HypnoTrust focuses on a controlled proof-of-concept audit workflow: frozen predictions, uncertainty-based review queue, calibration checks, locked/external evaluation framing, failure-mode analysis and an interpretable Streamlit dashboard.

## References to discuss in defense

- DeepSleepNet: automatic sleep stage scoring from raw single-channel EEG.
- SeqSleepNet: sequence-to-sequence automatic sleep staging.
- SleepTransformer: interpretability and uncertainty quantification for automatic sleep staging.
- U-PASS: uncertainty-guided pipeline for automated sleep staging.
