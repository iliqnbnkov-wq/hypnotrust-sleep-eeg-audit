---
applyTo: "experiments/**,results/**,docs/**,README.md,src/hypnotrust/calibration.py,src/hypnotrust/validation.py"
---
- Treat files under `experiments/` and `results/` as read-only evidence.
- State subject role, protocol and aggregation level next to every metric; keep participants, recordings and 30 s epochs distinct.
- SC409 has an evaluation history: never describe it as unused for future decisions.
- Calibration is fitted only on out-of-fold probabilities of SC400–SC408.
- Report uncertainty (spread across subjects) rather than only pooled values; do not claim improvement without a paired, subject-level comparison.
- Write NOT VERIFIED when evidence is absent.
