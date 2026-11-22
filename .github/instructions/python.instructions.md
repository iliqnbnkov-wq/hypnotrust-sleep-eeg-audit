---
applyTo: "**/*.py"
---
- Keep functions pure where possible; no hidden training, tuning or file writes in library code.
- Use type hints and `from __future__ import annotations` as in existing modules.
- Do not suppress warnings globally; fix or narrowly filter them.
- Temporary files must be removed after use; a lazily read MNE `Raw` (`preload=False`) must not outlive its file.
- Do not change feature order or the 13-feature baseline contract (`src/hypnotrust/constants.py`).
