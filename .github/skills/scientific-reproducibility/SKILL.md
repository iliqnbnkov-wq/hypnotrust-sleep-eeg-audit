---
name: scientific-reproducibility
description: Check whether a HypnoTrust result, table or claim can be traced to a file, protocol, version and aggregation level, and record what is NOT VERIFIED. Use when reviewing metrics, README claims or reproducibility notes.
---

# Scientific reproducibility check

1. Identify the claim and the exact source file (path and column or section).
2. Record experiment version, subject role (development, SC409, external holdout), protocol and aggregation (pooled epochs, per-subject mean, single subject).
3. Recompute only from frozen outputs already in the repository (for example `v5d_epoch_predictions.csv`); do not run models and do not evaluate SC409 or SC410–SC419 anew.
4. Compare with the documented value; report match, mismatch or NOT VERIFIED.
5. Note missing provenance (original commit, environment, absent CSV files). Never invent it.

Never modify frozen artifacts. Never claim a test or reproduction that was not executed.
