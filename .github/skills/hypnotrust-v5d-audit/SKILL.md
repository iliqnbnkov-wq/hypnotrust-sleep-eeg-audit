---
name: hypnotrust-v5d-audit
description: Audit the provenance, split definitions, metrics, and frozen outputs of the public HypnoTrust V5d evaluation without changing experimental artifacts.
---

# HypnoTrust V5d audit

## Use when

Use this skill when asked to inspect or summarize the public V5d frozen evaluation, verify its documentation against checked-in outputs, or identify gaps in its provenance.

## Checks

1. Read the V5d report, evaluation config, summary, predictions, confusion matrices, and error-pair outputs as needed.
2. Verify that subject roles are stated distinctly: SC400–SC408 development/training, SC409 historical locked evaluation, and SC410–SC419 external holdout.
3. Cross-check reported counts and metrics against the relevant checked-in data where possible. Identify the exact source for each value.
4. Distinguish documentation/output inspection from independently rerunning the evaluation. Do not imply the latter unless it was actually performed and approved.
5. Note missing provenance or unavailable inputs as unresolved.

## Constraints

- This is a read-only audit. Never edit, regenerate, delete, or overwrite frozen V5d outputs, scientific CSV/JSON results, trained models, historical results, or locked configurations.
- Never train, tune, calibrate, or run experiments on locked or external holdout subjects.
- Do not copy archived r4 materials into this repository.
- Do not invent results or treat a documented assertion as independently verified.

## Expected report

Report in Bulgarian with: scope and files inspected; verified split roles and values with evidence paths; discrepancies or unresolved checks; explicit statement whether results were independently reproduced; and confirmation that no frozen artifacts were changed.
