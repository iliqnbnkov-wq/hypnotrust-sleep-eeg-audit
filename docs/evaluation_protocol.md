# Evaluation Protocol Summary

HypnoTrust separates model development from locked and external evaluation.

Core principle:

- no training on locked test subject
- no tuning on locked test subject
- no calibration using locked test subject
- no feature selection using locked or external holdout subjects

The Streamlit dashboard reads frozen prediction outputs only.
