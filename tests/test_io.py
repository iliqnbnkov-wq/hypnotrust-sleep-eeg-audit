"""
Unit tests for hypnotrust.io.

All tests use temporary directories and never touch real baseline artifacts.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_valid_yaml(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "cfg.yaml", "model:\n  n_estimators: 500\n")
        self.assertEqual(load_config(path), {"model": {"n_estimators": 500}})

    def test_load_valid_yml_extension(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "cfg.yml", "key: value\n")
        self.assertEqual(load_config(path)["key"], "value")

    def test_empty_yaml_returns_empty_dict(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "empty.yaml", "")
        self.assertEqual(load_config(path), {})

    def test_yaml_null_returns_empty_dict(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "null.yaml", "null\n")
        self.assertEqual(load_config(path), {})

    def test_yaml_list_root_raises_value_error(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "list.yaml", "- item1\n- item2\n")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_yaml_scalar_root_raises_value_error(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "scalar.yaml", "42\n")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_missing_file_raises_file_not_found(self) -> None:
        from hypnotrust.io import load_config

        with self.assertRaises(FileNotFoundError):
            load_config(self.tmp / "missing.yaml")

    def test_unsupported_extension_raises_value_error(self) -> None:
        from hypnotrust.io import load_config

        path = _write(self.tmp / "cfg.toml", "[section]\nkey = 'val'\n")
        with self.assertRaises(ValueError):
            load_config(path)


class TestDataFrameIO(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_csv_roundtrip_preserves_columns_and_values(self) -> None:
        from hypnotrust.io import load_dataframe, save_dataframe

        original = pd.DataFrame(
            {
                "Subject": [0, 1, 2],
                "Stage": ["Wake", "N1", "REM"],
                "Fpz-Cz_Delta": [0.45, 0.31, 0.22],
            }
        )

        path = self.tmp / "test.csv"
        save_dataframe(original, path)
        loaded = load_dataframe(path)

        self.assertListEqual(list(loaded.columns), list(original.columns))
        self.assertListEqual(list(loaded["Subject"]), [0, 1, 2])
        self.assertListEqual(list(loaded["Stage"]), ["Wake", "N1", "REM"])

    def test_load_dataframe_missing_file_raises(self) -> None:
        from hypnotrust.io import load_dataframe

        with self.assertRaises(FileNotFoundError):
            load_dataframe(self.tmp / "missing.csv")

    def test_save_dataframe_wrong_extension_raises(self) -> None:
        from hypnotrust.io import save_dataframe

        dataframe = pd.DataFrame({"x": [1, 2]})

        with self.assertRaises(ValueError):
            save_dataframe(dataframe, self.tmp / "data.json")

    def test_load_dataframe_wrong_extension_raises(self) -> None:
        from hypnotrust.io import load_dataframe

        path = _write(self.tmp / "data.json", "[]")
        with self.assertRaises(ValueError):
            load_dataframe(path)

    def test_save_dataframe_creates_parent_dirs(self) -> None:
        from hypnotrust.io import load_dataframe, save_dataframe

        nested = self.tmp / "a" / "b" / "c" / "data.csv"
        dataframe = pd.DataFrame({"x": [1, 2]})

        save_dataframe(dataframe, nested)

        self.assertTrue(nested.exists())
        loaded = load_dataframe(nested)
        self.assertListEqual(list(loaded["x"]), [1, 2])


class TestModelIO(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_model_roundtrip(self) -> None:
        from hypnotrust.io import load_model, save_model

        original = {"n_estimators": 500, "random_state": 42}
        path = self.tmp / "model.joblib"

        save_model(original, path)
        loaded = load_model(path)

        self.assertEqual(loaded, original)

    def test_load_model_missing_file_raises(self) -> None:
        from hypnotrust.io import load_model

        with self.assertRaises(FileNotFoundError):
            load_model(self.tmp / "missing.joblib")

    def test_save_model_wrong_extension_raises(self) -> None:
        from hypnotrust.io import save_model

        with self.assertRaises(ValueError):
            save_model({}, self.tmp / "model.pkl")

    def test_load_model_wrong_extension_raises(self) -> None:
        from hypnotrust.io import load_model

        path = _write(self.tmp / "model.pkl", "dummy")
        with self.assertRaises(ValueError):
            load_model(path)

    def test_save_model_creates_parent_dirs(self) -> None:
        from hypnotrust.io import load_model, save_model

        nested = self.tmp / "d" / "e" / "model.joblib"
        original = {"key": "value"}

        save_model(original, nested)

        self.assertTrue(nested.exists())
        self.assertEqual(load_model(nested), original)


class TestMetricsIO(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_metrics_roundtrip(self) -> None:
        from hypnotrust.io import load_metrics, save_metrics

        original = {
            "accuracy": 0.7942,
            "kappa": 0.6920,
            "macro_f1": 0.7132,
        }

        path = self.tmp / "metrics.json"
        save_metrics(original, path)
        loaded = load_metrics(path)

        self.assertEqual(loaded, original)

    def test_metrics_are_sorted_on_disk(self) -> None:
        from hypnotrust.io import save_metrics

        path = self.tmp / "metrics.json"
        save_metrics({"z_score": 1, "accuracy": 0.9, "kappa": 0.8}, path)

        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(list(data.keys()), sorted(data.keys()))

    def test_metrics_indented_on_disk(self) -> None:
        from hypnotrust.io import save_metrics

        path = self.tmp / "metrics.json"
        save_metrics({"accuracy": 0.9}, path)

        raw = path.read_text(encoding="utf-8")
        self.assertIn("  ", raw)

    def test_load_metrics_missing_file_raises(self) -> None:
        from hypnotrust.io import load_metrics

        with self.assertRaises(FileNotFoundError):
            load_metrics(self.tmp / "missing.json")

    def test_load_metrics_non_dict_raises(self) -> None:
        from hypnotrust.io import load_metrics

        path = _write(self.tmp / "list.json", "[1, 2, 3]")
        with self.assertRaises(ValueError):
            load_metrics(path)

    def test_save_metrics_wrong_extension_raises(self) -> None:
        from hypnotrust.io import save_metrics

        with self.assertRaises(ValueError):
            save_metrics({"k": 1}, self.tmp / "metrics.txt")

    def test_load_metrics_wrong_extension_raises(self) -> None:
        from hypnotrust.io import load_metrics

        path = _write(self.tmp / "metrics.txt", '{"k": 1}')
        with self.assertRaises(ValueError):
            load_metrics(path)

    def test_save_metrics_creates_parent_dirs(self) -> None:
        from hypnotrust.io import save_metrics

        nested = self.tmp / "f" / "g" / "metrics.json"
        save_metrics({"accuracy": 0.79}, nested)

        self.assertTrue(nested.exists())

    def test_utf8_text_preserved(self) -> None:
        from hypnotrust.io import load_metrics, save_metrics

        original = {
            "описание": "точност на модела",
            "accuracy": 0.7942,
        }

        path = self.tmp / "metrics_utf8.json"
        save_metrics(original, path)

        raw = path.read_text(encoding="utf-8")
        self.assertIn("описание", raw)
        self.assertIn("точност на модела", raw)

        loaded = load_metrics(path)
        self.assertEqual(loaded["описание"], "точност на модела")


if __name__ == "__main__":
    unittest.main()
