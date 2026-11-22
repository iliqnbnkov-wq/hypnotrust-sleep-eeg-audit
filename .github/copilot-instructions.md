# HypnoTrust repository guidance

## Purpose and scientific limits

HypnoTrust is an educational/research proof of concept for inspecting sleep-stage predictions and prioritizing epochs for human review. It is not a clinical system, diagnostic tool, or replacement for a qualified sleep specialist. Do not describe it as clinically validated or state of the art.

## Evaluation provenance

- The V5d frozen external evaluation under `experiments/v5_expanded_cross_subject_evaluation/v5d_frozen_external_evaluation/` is the current public experimental workflow.
- The recorded V5d development/training subjects are SC400–SC408.
- SC409 (historically called Subject 9) is a historical locked evaluation subject with reported historical results; do not call it an untouched external holdout or conflate it with SC410–SC419.
- SC410–SC419 are the V5d external holdout subjects.
- Never use SC409 or SC410–SC419 for model training, hyperparameter optimization, feature selection, probability calibration, calibration-method selection, confidence-threshold selection, alpha selection, review-policy design, or model-version selection. Evaluation on these subjects is permitted only under the existing documented protocol.
- Do not retroactively claim that a previously inspected test set was never used for development decisions. Claim an untouched holdout only when the documented protocol and evidence support that specific claim.
- Keep training, development decisions, calibration, and final evaluation strictly separated according to the recorded protocol.
- Treat frozen V5d outputs, scientific result CSV/JSON files, trained models, historical evaluation artifacts, and locked experiment configurations as read-only. Never regenerate, overwrite, remove, or otherwise alter them.
- Do not copy materials from an archived r4 workspace into this repository.

## Evidence and reporting

- Never invent metrics, test outcomes, experiment results, citations, or scientific conclusions.
- Tie each factual conclusion to inspectable repository evidence (file, section, data field, or command output); distinguish documented/reported values from values independently reproduced in the current work.
- If an assertion cannot be checked from available evidence, label it as unresolved rather than presenting it as fact.
- Do not claim tests, builds, or experiments passed unless they were actually run and their output was checked.
- Preserve existing measured results when editing documentation; do not broaden their interpretation.

## Software and collaboration

- Follow the existing Python, Streamlit, and unittest patterns. The dashboard is an inspection tool and must not train or tune models as a side effect.
- Avoid commands that install dependencies or mutate data unless explicitly approved. In particular, `run_app.ps1` and `run_tests.ps1` install/update packages; do not use them for validation without approval.
- Provide responses and technical explanations in Bulgarian. Keep code identifiers and standard tooling terms in their conventional form.
- Make focused changes and report files changed, checks actually performed, and unresolved limitations.
