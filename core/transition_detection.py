"""
Transition Detection Module for HypnoTrust v2.

Detects N2↔N3 transitions from expert hypnogram annotations using the k-stability criterion.

Based on E0 methodology: transitions are changes between adjacent 30-second scored epochs.
k-stability criterion requires k consecutive epochs in the previous stage and k consecutive epochs in the new stage.

RESEARCH PROTOTYPE ONLY. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionEvent:
    """
    Represents a single N2↔N3 transition event.

    Attributes:
        epoch_index: Index of the boundary epoch (0-based)
        kind: Transition direction ("N2->N3" or "N3->N2")
        before_stage: Stage before the boundary
        after_stage: Stage after the boundary
        boundary_seconds: Time of the boundary in seconds from recording start
        k: Stability criterion used
    """

    epoch_index: int
    kind: str
    before_stage: str
    after_stage: str
    boundary_seconds: float
    k: int


def detect_transitions(
    labels: list[str],
    epoch_length_sec: float = 30.0,
    k: int = 3,
) -> list[TransitionEvent]:
    """
    Detect N2↔N3 transitions using k-stability criterion.

    Args:
        labels: List of stage labels from expert hypnogram
        epoch_length_sec: Duration of each epoch in seconds
        k: Stability criterion (number of consecutive epochs required)

    Returns:
        List of TransitionEvent objects

    Notes:
        - A transition is a change between adjacent epochs
        - k-stability requires k consecutive epochs in the previous stage before the boundary
        - and k consecutive epochs in the new stage after the boundary
        - This filters out brief stage fluctuations
        - Boundaries are 30-s label boundaries, not precise physiological transition moments
    """
    if not labels:
        return []

    n_epochs = len(labels)
    events = []

    # Look for N2↔N3 boundaries
    for i in range(n_epochs - 1):
        current = labels[i]
        next_label = labels[i + 1]

        # Check if this is a N2↔N3 transition
        is_n2_to_n3 = current == "N2" and next_label == "N3"
        is_n3_to_n2 = current == "N3" and next_label == "N2"

        if not (is_n2_to_n3 or is_n3_to_n2):
            continue

        # Check k-stability criterion
        # Need k consecutive epochs in current stage before the boundary (including epoch i)
        if i + 1 >= k:
            before_window = labels[i - k + 1 : i + 1]
            if not all(label == current for label in before_window):
                continue
        else:
            # Not enough epochs before the boundary
            continue

        # Need k consecutive epochs in next stage after the boundary (epochs i+1 to i+k)
        if i + 1 + k <= n_epochs:
            after_window = labels[i + 1 : i + k + 1]
            if not all(label == next_label for label in after_window):
                continue
        else:
            # Not enough epochs after the boundary
            continue

        # All checks passed - this is a valid transition
        kind = "N2->N3" if is_n2_to_n3 else "N3->N2"
        boundary_seconds = (i + 1) * epoch_length_sec  # Boundary is between epochs i and i+1

        events.append(
            TransitionEvent(
                epoch_index=i + 1,  # Boundary epoch index
                kind=kind,
                before_stage=current,
                after_stage=next_label,
                boundary_seconds=boundary_seconds,
                k=k,
            )
        )

    return events


def count_transitions_by_kind(
    events: list[TransitionEvent],
) -> dict[str, int]:
    """
    Count transitions by kind.

    Returns:
        Dictionary with counts for "N2->N3", "N3->N2", and "BOTH"
    """
    n2_to_n3 = sum(1 for e in events if e.kind == "N2->N3")
    n3_to_n2 = sum(1 for e in events if e.kind == "N3->N2")

    return {
        "N2->N3": n2_to_n3,
        "N3->N2": n3_to_n2,
        "BOTH": n2_to_n3 + n3_to_n2,
    }


def filter_transitions_by_kind(
    events: list[TransitionEvent],
    kind: str,
) -> list[TransitionEvent]:
    """
    Filter transitions by kind.

    Args:
        events: List of TransitionEvent objects
        kind: "N2->N3", "N3->N2", or "BOTH"

    Returns:
        Filtered list of events
    """
    if kind == "BOTH":
        return events
    return [e for e in events if e.kind == kind]
