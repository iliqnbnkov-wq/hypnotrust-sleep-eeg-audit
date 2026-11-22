---
name: eeg-pipeline-audit
description: Audit the HypnoTrust EEG/EOG pipeline (EDF loading, features, temporal context, calibration, review queue) for leakage, alignment and resource-handling defects without retraining.
---

# EEG pipeline audit

Check, with evidence (file and line):

- Channel resolution and epoch alignment between PSG and hypnogram (30 s epochs, trailing annotation overhang).
- Subject-wise separation and absence of SC409 / SC410–SC419 in any fitting step.
- Temporal context does not cross recording or segment boundaries incorrectly.
- Calibration is out-of-fold on development subjects only.
- Lazy MNE objects versus deleted temporary files; temporary files removed.
- Inference code does not train, tune or write to frozen artifacts.

Prefer small reproducible checks on synthetic data. Report findings as factual error, missing evidence or editorial. Do not change thresholds, models or labels.
