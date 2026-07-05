"""Shared pytest configuration for HypnoTrust."""

from hypnotrust.constants import BASELINE_FEATURE_COLUMNS


def pytest_configure(config) -> None:
    """Fail early if the locked baseline feature contract is broken."""
    if len(BASELINE_FEATURE_COLUMNS) != 13:
        raise RuntimeError(
            "Baseline v1 must contain exactly 13 feature columns."
        )
