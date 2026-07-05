"""Locked constants for HypnoTrust Baseline v1."""

from typing import Final


BASELINE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "Fpz-Cz_Delta",
    "Fpz-Cz_Theta",
    "Fpz-Cz_Alpha",
    "Fpz-Cz_Sigma",
    "Fpz-Cz_Beta",
    "Pz-Oz_Delta",
    "Pz-Oz_Theta",
    "Pz-Oz_Alpha",
    "Pz-Oz_Sigma",
    "Pz-Oz_Beta",
    "EOG_Robust_Range",
    "EOG_Diff_RMS",
    "EOG_Relative_Low_Power",
)

METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "Subject",
    "Recording",
    "Epoch",
    "Stage",
)

EEG_CHANNELS: Final[tuple[str, ...]] = (
    "Fpz-Cz",
    "Pz-Oz",
)

EOG_CHANNEL: Final[str] = "horizontal"

ANALYSIS_CHANNELS: Final[tuple[str, ...]] = (
    "Fpz-Cz",
    "Pz-Oz",
    "horizontal",
)

ANNOTATION_TO_EVENT: Final[dict[str, int]] = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

EVENT_ID: Final[dict[str, int]] = {
    "Wake": 1,
    "N1": 2,
    "N2": 3,
    "N3": 4,
    "REM": 5,
}

ID_TO_STAGE: Final[dict[int, str]] = {
    1: "Wake",
    2: "N1",
    3: "N2",
    4: "N3",
    5: "REM",
}

STAGE_ORDER: Final[tuple[str, ...]] = (
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM",
)

DEVELOPMENT_SUBJECTS: Final[tuple[int, ...]] = tuple(range(9))

EXCLUDED_TEST_SUBJECTS: Final[tuple[int, ...]] = (9,)

FINAL_TEST_SUBJECT: Final[int] = 9
