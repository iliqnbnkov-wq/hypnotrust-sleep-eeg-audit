---
name: hypnotrust-python-tests
description: Assess and run HypnoTrust Python tests safely, without automatic dependency installation or model experiments.
---

# HypnoTrust Python tests

## Use when

Use this skill when checking test coverage, dependencies, test readiness, or validating Python code changes in this repository.

## Checks

1. Inspect test files, test configuration, dependency manifests, and test fixtures before selecting a command.
2. Identify whether the command may install packages, create environments, download data, train models, or modify files.
3. Prefer a focused unittest command for the affected tests; use the full suite only when appropriate.
4. Check dependency availability without installing packages when installation has not been approved.
5. Report exact commands and their actual output/status. If dependencies are missing, stop and report them.

## Constraints

- Never use `run_tests.ps1` or `run_app.ps1` as a validation shortcut: both install/update dependencies.
- Do not install dependencies, create/modify datasets, train models, or run scientific experiments without explicit approval.
- Before running unit tests, present the exact command and obtain user approval when required by the request.
- Do not modify frozen V5d outputs, scientific result CSV/JSON files, trained models, historical artifacts, or locked configurations.

## Expected report

Report in Bulgarian with: test/config/dependency files inspected; proposed exact command; whether approval was obtained; checks actually performed and results; missing dependencies or limitations; and confirmation that no packages or protected data were changed.
