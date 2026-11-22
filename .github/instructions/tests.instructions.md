---
applyTo: "tests/**/*.py"
---
- Use `assert`, not `return True/False`, in pytest functions.
- Tests must be deterministic and run without Sleep-EDF data, trained-model training or network access; use synthetic data or `pytest.importorskip` / `skipif` with a clear reason.
- Tests that need SC4001 or other external data are marked and excluded from CI.
- A skipped or early-returning test is not a pass for the behaviour it names; say so in reports.
