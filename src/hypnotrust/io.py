"""
Safe input/output utilities for HypnoTrust.

The module handles YAML configuration files, CSV data tables,
joblib model artifacts, and JSON metric files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


def _require_exists(path: Path) -> None:
    """Raise FileNotFoundError when the requested file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _require_suffix(path: Path, expected: str) -> None:
    """Raise ValueError when the file has an unexpected extension."""
    if path.suffix.lower() != expected.lower():
        raise ValueError(
            f"Expected a '{expected}' file, "
            f"got '{path.suffix}': {path}"
        )


def load_config(path: Path) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file with a ``.yaml`` or ``.yml`` extension.

    Returns
    -------
    dict
        Parsed YAML mapping. Empty YAML documents return an empty dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is unsupported or the YAML root is not a dict.
    """
    path = Path(path)
    _require_exists(path)

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported extension '{path.suffix}'; "
            f"expected .yaml or .yml: {path}"
        )

    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            "YAML root must be a mapping "
            f"(dict), got {type(data).__name__}: {path}"
        )

    return data


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save a pandas DataFrame as a UTF-8 CSV file.

    Parent directories are created automatically.

    Raises
    ------
    ValueError
        If the destination does not use the ``.csv`` extension.
    """
    path = Path(path)
    _require_suffix(path, ".csv")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a UTF-8 CSV file as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not use the ``.csv`` extension.
    """
    path = Path(path)
    _require_exists(path)
    _require_suffix(path, ".csv")

    return pd.read_csv(
        path,
        encoding="utf-8",
    )


def save_model(
    model: Any,
    path: Path,
) -> None:
    """Save a Python object as a joblib artifact.

    Parent directories are created automatically.

    Raises
    ------
    ValueError
        If the destination does not use the ``.joblib`` extension.
    """
    path = Path(path)
    _require_suffix(path, ".joblib")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )


def load_model(path: Path) -> Any:
    """Load an object from a joblib artifact.

    Warning
    -------
    Only load joblib files from trusted sources.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not use the ``.joblib`` extension.
    """
    path = Path(path)
    _require_exists(path)
    _require_suffix(path, ".joblib")

    return joblib.load(path)


def save_metrics(
    metrics: dict,
    path: Path,
) -> None:
    """Save a metrics dictionary as formatted UTF-8 JSON.

    Parent directories are created automatically.

    Raises
    ------
    ValueError
        If the destination does not use the ``.json`` extension.
    """
    path = Path(path)
    _require_suffix(path, ".json")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_metrics(path: Path) -> dict:
    """Load a metrics dictionary from a UTF-8 JSON file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is not ``.json`` or the JSON root is not a dict.
    """
    path = Path(path)
    _require_exists(path)
    _require_suffix(path, ".json")

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root must be an object "
            f"(dict), got {type(data).__name__}: {path}"
        )

    return data