---
name: hypnotrust-streamlit-review
description: Review HypnoTrust's Streamlit dashboard for consistency with its frozen-prediction and research-only behavior.
---

# HypnoTrust Streamlit review

## Use when

Use this skill when reviewing or changing `app/app.py`, dashboard behavior, or user-facing descriptions of predictions, confidence, uncertainty, review queues, and evaluation results.

## Checks

1. Trace the relevant dashboard behavior to its `core/` helpers and configured artifacts.
2. Confirm the dashboard does not train, tune, calibrate, or overwrite model/evaluation artifacts.
3. Check labels, class order, subject/evaluation roles, uncertainty interpretation, and exports against their data sources.
4. Check missing-file and invalid-input behavior; errors should be visible rather than silently converted into plausible results.
5. If code changes are requested, use focused tests for the changed behavior and report checks actually run.

## Constraints

- Do not change frozen V5d outputs, scientific result CSV/JSON files, trained models, historical evaluation artifacts, or locked configurations.
- Do not present confidence as correctness, clinical reliability, or a clinical recommendation.
- Do not use app launch scripts for validation without approval: they install/update dependencies.
- Do not install dependencies or run training/experiments without explicit approval.

## Expected report

Report in Bulgarian with: files and behavior reviewed; verified data flow; issues and evidence; changes/tests if applicable; and explicit confirmation that no protected artifact was modified.
