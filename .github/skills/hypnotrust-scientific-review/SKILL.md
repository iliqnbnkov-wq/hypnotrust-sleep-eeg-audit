---
name: hypnotrust-scientific-review
description: Review HypnoTrust scientific wording for evidence, evaluation leakage, provenance, and appropriate limits without changing measured results.
---

# HypnoTrust scientific review

## Use when

Use this skill to review README, model-card, evaluation, experiment, or related-work claims; assess whether claims have direct supporting evidence; or propose more cautious wording.

## Checks

1. Map each material claim to a direct report, configuration, result artifact, test, or citation.
2. Separate reported metrics from results independently reproduced in the current work.
3. Check that SC409 is identified as the historical locked evaluation and SC410–SC419 as the V5d external holdout.
4. Check for leakage risks or ambiguity among training, tuning, calibration, policy selection, and final evaluation.
5. Check dataset, annotation, uncertainty, calibration, selective-prediction, and related-work statements for appropriate caveats.
6. Flag claims that lack evidence, need citations, or exceed the available study design.

## Constraints

- Do not alter frozen outputs, measured result CSV/JSON files, trained models, historical artifacts, or locked configurations.
- Do not fabricate citations, metrics, statistical significance, clinical utility, novelty, or validation.
- Do not infer clinical readiness or broad generalization from a limited proof of concept.
- Do not copy r4 archive files into the public repository.

## Expected report

Report in Bulgarian with: claim; supporting evidence and path (or “not located”); evidence strength/limitation; suggested bounded wording; and unresolved issues. State whether any numerical result was independently reproduced.
