---
applyTo: "app/**/*.py"
---
- The dashboard is for inspection and inference only: no model training or tuning.
- Uploaded files are untrusted: delete temporary copies, never keep uploaded EDF bytes on disk after processing, and do not unpickle user-supplied files.
- Prefer current Streamlit parameters (`width="stretch"` instead of `use_container_width`) when changing a call site and verify in the pinned Streamlit version.
- Do not hide warnings with `warnings.filterwarnings("ignore")`.
- User mode shows predictions only; accuracy and kappa appear only in Research mode with verified labels.
