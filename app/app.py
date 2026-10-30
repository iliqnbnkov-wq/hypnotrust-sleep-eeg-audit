# app/app.py
"""
HypnoTrust — Research Prototype Dashboard
Local Streamlit application for sleep-stage classification and uncertainty review.

Run from project root:
    $env:PYTHONPATH = "$PWD;$PWD\\src"
    python -m streamlit run .\\app\\app.py

IMPORTANT:
Research prototype only.
Not for clinical diagnosis.
Not a replacement for a qualified sleep specialist.
"""

from __future__ import annotations

import warnings
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HypnoTrust",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Core imports — hard failure: app only runs inside project environment
# ---------------------------------------------------------------------------

try:
    from core.config import get_default_pipeline_config, validate_app_paths
    from core.pipeline import run_feature_dataframe_pipeline, run_full_pipeline
    from core.review_queue import build_review_queue
    from core.reporter import predictions_to_csv_bytes, review_queue_to_csv_bytes
    from core.edf_evaluation import (
        evaluate_edf_predictions,
        export_evaluation_csv,
        export_evaluation_json,
        validate_annotation_alignment,
    )
    from core.transition_detection import detect_transitions, filter_transitions_by_kind
    from core.spectral_analysis import compute_spectral_windows, compute_before_after_power
    from core.eeg_extraction import extract_eeg_segment
except ImportError as _err:
    st.error(
        f"**Core module import failed:** `{_err}`\n\n"
        "Launch the app from the project root with:\n"
        "```powershell\n"
        "$env:PYTHONPATH = \"$PWD;$PWD\\src\"\n"
        "python -m streamlit run .\\app\\app.py\n"
        "```"
    )
    st.stop()

try:
    from core.metrics import (
        compute_metrics_if_available,
        build_confusion_matrix_df,
        build_per_stage_metrics_df,
    )

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

try:
    from hypnotrust.temporal import build_temporal_feature_dataframe

    _TEMPORAL_BUILDER_AVAILABLE = True
except ImportError:
    _TEMPORAL_BUILDER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Load pipeline config once — drives feature schema everywhere
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_cfg():
    return get_default_pipeline_config()


try:
    _CFG = _load_cfg()
    _TEMPORAL_FEATURE_COLUMNS: list[str] = list(_CFG.feature_columns)
except Exception as _cfg_err:
    st.error(
        f"**Pipeline config failed to load:** `{_cfg_err}`\n\n"
        "Check `core/config.py`, the temporal v2 model artifact, "
        "and the temporal v2 features JSON."
    )
    st.stop()

try:
    _PATH_PROBLEMS = validate_app_paths()
except Exception as _path_err:
    _PATH_PROBLEMS = [f"Path validation failed: {_path_err}"]

if _PATH_PROBLEMS:
    st.error("**Configuration problem:**")
    for _problem in _PATH_PROBLEMS:
        st.write("-", _problem)
    st.stop()


_BASELINE_FEATURE_COLUMNS: list[str] = [
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
]


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #161b22;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}

.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(46,160,67,0.35);
}

[data-testid="stDownloadButton"] > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #79c0ff;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.4rem 0.9rem;
    transition: all 0.15s;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #30363d;
    border-color: #58a6ff;
}

[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 10px;
    padding: 1rem;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #21262d;
    overflow: hidden;
}

hr {
    border-color: #21262d;
    margin: 1rem 0;
}

code {
    color: #79c0ff !important;
    background-color: #161b22 !important;
    border-radius: 4px;
    padding: 0.05rem 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

STAGE_COLORS = {
    "Wake": "#f78166",
    "REM": "#79c0ff",
    "N1": "#d2a8ff",
    "N2": "#7ee787",
    "N3": "#56d364",
}

STAGE_Y = {
    "Wake": 5,
    "REM": 4,
    "N1": 3,
    "N2": 2,
    "N3": 1,
}

STAGE_ORDER = ("Wake", "N1", "N2", "N3", "REM")

PLOTLY_BASE = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=12, r=12, t=40, b=12),
)

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _card(
    title: str,
    value: str,
    sub: str = "",
    color: str = "#58a6ff",
) -> str:
    return (
        f'<div style="background:#161b22;border:1px solid #21262d;'
        f'border-radius:12px;padding:1.1rem 1.3rem;text-align:center;'
        f'border-top:3px solid {color};">'
        f'<div style="color:#8b949e;font-size:0.74rem;font-weight:600;'
        f'letter-spacing:0.06em;text-transform:uppercase;'
        f'margin-bottom:0.4rem;">{title}</div>'
        f'<div style="font-size:1.8rem;font-weight:700;color:#e6edf3;'
        f'line-height:1.1;">{value}</div>'
        + (
            f'<div style="color:#6e7681;font-size:0.76rem;margin-top:0.3rem;">'
            f'{sub}</div>'
            if sub
            else ""
        )
        + "</div>"
    )


def _section_header(
    text: str,
    badge: str = "",
    badge_color: str = "#388bfd",
) -> None:
    badge_html = (
        f'<span style="background:{badge_color}22;color:{badge_color};'
        f'border:1px solid {badge_color}44;border-radius:20px;'
        f'font-size:0.68rem;font-weight:600;padding:0.12rem 0.55rem;'
        f'margin-left:0.6rem;letter-spacing:0.05em;">{badge}</span>'
        if badge
        else ""
    )

    st.markdown(
        f'<h3 style="color:#e6edf3;font-size:1.02rem;font-weight:600;'
        f'margin:1.4rem 0 0.7rem;padding-bottom:0.45rem;'
        f'border-bottom:1px solid #21262d;">{text}{badge_html}</h3>',
        unsafe_allow_html=True,
    )


def _disclaimer() -> None:
    st.markdown(
        '<div style="background:#161b22;border:1px solid #f7871622;'
        'border-left:3px solid #f78716;border-radius:8px;'
        'padding:0.55rem 1rem;margin-bottom:1rem;">'
        '<span style="color:#f78716;font-weight:600;font-size:0.79rem;">'
        '⚠ RESEARCH PROTOTYPE</span>'
        '<span style="color:#8b949e;font-size:0.77rem;margin-left:0.6rem;">'
        'Not for clinical use. Results do not constitute medical diagnosis '
        'and must be reviewed by a qualified sleep specialist.</span></div>',
        unsafe_allow_html=True,
    )


def _mode_badge(mode: str) -> None:
    palette = {
        "Demo": ("#388bfd", "🎯  DEMO"),
        "Research": ("#a371f7", "🔬  RESEARCH"),
        "User": ("#3fb950", "👤  USER"),
        "EDF Upload": ("#f78716", "🧪  EDF"),
        "Transition Explorer": ("#d2a8ff", "🌊 TRANSITION"),
    }

    color, label = palette.get(mode, ("#8b949e", mode.upper()))

    st.sidebar.markdown(
        f'<div style="background:{color}18;border:1px solid {color}44;'
        f'border-radius:8px;padding:0.4rem 0.7rem;text-align:center;'
        f'font-size:0.73rem;font-weight:700;color:{color};'
        f'letter-spacing:0.08em;margin-bottom:0.5rem;">{label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Feature DataFrame validation / upgrade
# ---------------------------------------------------------------------------

def _extract_verified_labels(df: pd.DataFrame) -> list[str] | None:
    """
    Extract labels only when a Stage column is complete and has at least 2 classes.

    This prevents accidental metrics from placeholder or incomplete labels.
    """

    if "Stage" not in df.columns:
        return None

    if df["Stage"].isna().any():
        return None

    labels = df["Stage"].astype(str).tolist()

    if len(set(labels)) < 2:
        return None

    return labels


def _resolve_feature_df(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[str] | None, str | None, str | None]:
    """
    Resolve uploaded CSV into temporal v2 features.

    Returns:
        feature_df, labels, error_message, schema_name

    Supported input schemas:
    1. Temporal v2 CSV:
       contains all cfg.feature_columns.
    2. Baseline v1 CSV:
       contains 13 baseline features + Subject, Recording, Epoch.
       Stage is optional for inference. If missing, a placeholder Stage is
       inserted only to satisfy build_temporal_feature_dataframe().
    """

    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return None, None, "Uploaded CSV is empty or invalid.", None

    # Path 1: already temporal v2 features.
    missing_temporal = [
        col for col in _TEMPORAL_FEATURE_COLUMNS
        if col not in raw_df.columns
    ]

    if not missing_temporal:
        feature_df = raw_df[_TEMPORAL_FEATURE_COLUMNS].copy()
        labels = _extract_verified_labels(raw_df)
        return feature_df, labels, None, "temporal_v2"

    # Path 2: baseline v1 features -> temporal v2 expansion.
    missing_baseline = [
        col for col in _BASELINE_FEATURE_COLUMNS
        if col not in raw_df.columns
    ]

    if not missing_baseline:
        if not _TEMPORAL_BUILDER_AVAILABLE:
            return (
                None,
                None,
                "`hypnotrust.temporal.build_temporal_feature_dataframe` "
                "is not importable. Cannot auto-expand baseline CSV. "
                "Please upload a temporal v2 feature CSV.",
                None,
            )

        required_meta = ["Subject", "Recording", "Epoch"]
        missing_meta = [
            col for col in required_meta
            if col not in raw_df.columns
        ]

        if missing_meta:
            return (
                None,
                None,
                f"Baseline CSV detected, but metadata columns are missing: "
                f"{missing_meta}. Add Subject, Recording, Epoch, or upload "
                f"a temporal v2 feature CSV.",
                None,
            )

        working_df = raw_df.copy()

        # Stage is metadata needed by build_temporal_feature_dataframe.
        # It is NOT used as a model feature. For inference-only baseline CSVs,
        # a safe placeholder is allowed.
        real_labels_available = _extract_verified_labels(working_df) is not None

        if "Stage" not in working_df.columns:
            working_df["Stage"] = "Wake"
        else:
            working_df["Stage"] = working_df["Stage"].fillna("Wake").astype(str)

        try:
            temporal_df = build_temporal_feature_dataframe(working_df)
        except Exception as exc:
            return None, None, f"Temporal feature building failed: {exc}", None

        missing_after_build = [
            col for col in _TEMPORAL_FEATURE_COLUMNS
            if col not in temporal_df.columns
        ]

        if missing_after_build:
            return (
                None,
                None,
                f"Temporal feature building finished, but expected columns "
                f"are still missing. First missing columns: "
                f"{missing_after_build[:5]}",
                None,
            )

        feature_df = temporal_df[_TEMPORAL_FEATURE_COLUMNS].copy()

        labels = None
        if real_labels_available:
            labels = _extract_verified_labels(temporal_df)

        return feature_df, labels, None, "baseline_v1_expanded"

    # Path 3: unknown schema.
    return (
        None,
        None,
        "Unrecognised CSV schema.\n\n"
        f"Expected one of:\n"
        f"- Temporal v2 CSV with {len(_TEMPORAL_FEATURE_COLUMNS)} feature columns "
        f"such as `{_TEMPORAL_FEATURE_COLUMNS[0]}` … "
        f"`{_TEMPORAL_FEATURE_COLUMNS[-1]}`.\n"
        f"- Baseline CSV with 13 baseline feature columns such as "
        f"`{_BASELINE_FEATURE_COLUMNS[0]}` … "
        f"`{_BASELINE_FEATURE_COLUMNS[-1]}` plus Subject, Recording, Epoch. "
        f"Stage is optional for prediction but required for metrics.\n\n"
        f"Missing temporal columns, first 5: {missing_temporal[:5]}\n"
        f"Missing baseline columns, first 5: {missing_baseline[:5]}",
        None,
    )


# ---------------------------------------------------------------------------
# Synthetic demo feature generator
# ---------------------------------------------------------------------------

_DEMO_PROFILES = {
    "Healthy adult": {
        "sequence": [
            "Wake",
            "N1",
            "N2",
            "N3",
            "N3",
            "N2",
            "REM",
            "N2",
            "N3",
            "N2",
            "REM",
            "N2",
            "N2",
            "REM",
            "N1",
            "Wake",
        ],
    },
    "Fragmented sleep": {
        "sequence": [
            "Wake",
            "Wake",
            "N1",
            "N2",
            "Wake",
            "N1",
            "N2",
            "N3",
            "Wake",
            "N2",
            "REM",
            "Wake",
            "N1",
            "N2",
            "N2",
            "Wake",
        ],
    },
    "REM-dominant": {
        "sequence": [
            "Wake",
            "N1",
            "N2",
            "REM",
            "REM",
            "REM",
            "N2",
            "REM",
            "REM",
            "N2",
            "REM",
            "REM",
            "N1",
            "Wake",
            "Wake",
            "Wake",
        ],
    },
    "Deep sleep emphasis": {
        "sequence": [
            "Wake",
            "N1",
            "N2",
            "N3",
            "N3",
            "N3",
            "N3",
            "N2",
            "N3",
            "N3",
            "N2",
            "REM",
            "N2",
            "N3",
            "N2",
            "Wake",
        ],
    },
}

_STAGE_DELTA = {
    "Wake": 0.18,
    "N1": 0.28,
    "N2": 0.38,
    "N3": 0.62,
    "REM": 0.25,
}


def _make_demo_feature_df(
    profile_name: str,
    n_epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate a synthetic DataFrame with exactly cfg.feature_columns.

    The generated values are only for app demonstration.
    The model still runs through the real locked temporal v2 pipeline.
    """

    rng = np.random.default_rng(seed)
    sequence = _DEMO_PROFILES[profile_name]["sequence"]
    stages = [sequence[i % len(sequence)] for i in range(n_epochs)]

    rows: list[dict[str, float]] = []

    for epoch_idx, stage in enumerate(stages):
        delta_base = _STAGE_DELTA.get(stage, 0.35)
        row: dict[str, float] = {}

        for column in _TEMPORAL_FEATURE_COLUMNS:
            column_lower = column.lower()

            if "delta" in column_lower:
                value = delta_base + rng.normal(0, 0.04)
            elif "theta" in column_lower:
                value = 0.25 - delta_base * 0.15 + rng.normal(0, 0.03)
            elif "alpha" in column_lower:
                value = 0.15 + rng.normal(0, 0.03)
            elif "sigma" in column_lower:
                value = 0.12 + rng.normal(0, 0.02)
            elif "beta" in column_lower:
                value = 0.10 + rng.normal(0, 0.02)
            elif "robust_range" in column_lower:
                value = float(np.clip(rng.exponential(0.3), 0, 2))
            elif "diff_rms" in column_lower:
                value = float(np.clip(rng.exponential(0.2), 0, 1.5))
            elif "low_power" in column_lower or "relative_low" in column_lower:
                value = 0.35 + rng.normal(0, 0.08)
            else:
                value = rng.normal(0.3, 0.12)

            if any(
                tag in column_lower
                for tag in ("t-", "t+", "lag", "diff")
            ):
                value += rng.normal(0, 0.03)

            row[column] = float(np.clip(value, -2.0, 2.0))

        rows.append(row)

    feature_df = pd.DataFrame(rows, columns=_TEMPORAL_FEATURE_COLUMNS)

    return feature_df, stages


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    feature_df: pd.DataFrame,
    labels: list[str] | None,
    review_fraction: float,
    mode: str,
):
    """
    Call the real backend pipeline and build the review queue.
    """

    try:
        with st.spinner("Running prediction pipeline…"):
            result = run_feature_dataframe_pipeline(
                feature_df=feature_df,
                labels=labels,
                config=_CFG,
            )
    except Exception as exc:
        st.error(f"**Pipeline error:** {exc}")
        return None, None

    try:
        queue = build_review_queue(
            uncertainty_df=result.uncertainty_df,
            predicted_stages=result.predicted_stages,
            fraction=review_fraction,
            labels=result.labels if mode == "Research" else None,
        )
    except Exception as exc:
        st.error(f"**Review queue error:** {exc}")
        return result, None

    return result, queue


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _plot_synchronized_hypnogram(
    true_stages: list[str],
    predicted_stages: list[str],
    confidences: list[float] | None = None,
    highlight_errors: bool = True,
) -> go.Figure:
    """
    Plot synchronized hypnogram with true and predicted stages on the same axis.

    Args:
        true_stages: Ground-truth stage labels
        predicted_stages: Predicted stage labels
        confidences: Optional confidence values for color-coding
        highlight_errors: Whether to highlight prediction errors
    """
    n = len(predicted_stages)
    x = list(range(n))
    y_true = [STAGE_Y.get(stage, 0) for stage in true_stages]
    y_pred = [STAGE_Y.get(stage, 0) for stage in predicted_stages]

    fig = go.Figure()

    # True stages (dotted line) - single trace to avoid duplicate legend
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_true,
            mode="lines",
            name="Verified Label",
            line=dict(
                color="#6e7681",
                width=2.0,
                dash="dot",
            ),
            opacity=0.7,
            hovertemplate="Epoch %{x}: True=%{y}<extra></extra>",
        )
    )

    # Predicted stages (solid line)
    if highlight_errors:
        # Color segments based on correctness
        errors = [true != pred for true, pred in zip(true_stages, predicted_stages)]

        # Draw correct predictions in green
        for i in range(max(n - 1, 0)):
            if not errors[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i], x[i + 1]],
                        y=[y_pred[i], y_pred[i + 1]],
                        mode="lines",
                        showlegend=False,
                        line=dict(
                            color="#3fb950",
                            width=2.5,
                        ),
                        hovertemplate=f"Epoch {x[i]}: True={true_stages[i]}, Pred={predicted_stages[i]}, Conf={confidences[i]:.3f if confidences and i < len(confidences) else 'N/A'} (Correct)<extra></extra>",
                    )
                )

        # Draw errors in red
        for i in range(max(n - 1, 0)):
            if errors[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i], x[i + 1]],
                        y=[y_pred[i], y_pred[i + 1]],
                        mode="lines",
                        showlegend=False,
                        line=dict(
                            color="#f78166",
                            width=3.0,
                        ),
                        hovertemplate=f"Epoch {x[i]}: True={true_stages[i]}, Pred={predicted_stages[i]}, Conf={confidences[i]:.3f if confidences and i < len(confidences) else 'N/A'} (ERROR)<extra></extra>",
                    )
                )
    else:
        # Draw all predictions in one color
        for i in range(max(n - 1, 0)):
            stage = predicted_stages[i]
            fig.add_trace(
                go.Scatter(
                    x=[x[i], x[i + 1]],
                    y=[y_pred[i], y_pred[i + 1]],
                    mode="lines",
                    showlegend=False,
                    line=dict(
                        color=STAGE_COLORS.get(stage, "#8b949e"),
                        width=2.5,
                    ),
                    hovertemplate=f"Epoch {x[i]}: {stage}<extra></extra>",
                )
            )

    # Legend entries (single entry for Verified Label, no duplicates)
    if highlight_errors:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Correct Prediction",
                line=dict(color="#3fb950", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Prediction Error",
                line=dict(color="#f78166", width=3.0),
            )
        )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Synchronized Hypnogram: Verified vs Predicted",
            font=dict(size=13, color="#8b949e"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(STAGE_Y.values()),
            ticktext=list(STAGE_Y.keys()),
            range=[0.3, 5.7],
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        xaxis=dict(
            title="Epoch",
            gridcolor="#21262d",
            linecolor="#30363d",
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="#161b22"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="#161b22",
            bordercolor="#21262d",
            borderwidth=1,
        ),
        height=320,
    )

    return fig


def _plot_hypnogram(
    stages: list[str],
    true_stages: list[str] | None = None,
) -> go.Figure:
    n = len(stages)
    x = list(range(n))
    y_pred = [STAGE_Y.get(stage, 0) for stage in stages]

    fig = go.Figure()

    if true_stages:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[STAGE_Y.get(stage, 0) for stage in true_stages],
                mode="lines",
                name="Verified label",
                line=dict(
                    color="#6e7681",
                    width=1.5,
                    dash="dot",
                ),
                opacity=0.5,
            )
        )

    for i in range(max(n - 1, 0)):
        stage = stages[i]
        fig.add_trace(
            go.Scatter(
                x=[x[i], x[i + 1]],
                y=[y_pred[i], y_pred[i + 1]],
                mode="lines",
                showlegend=False,
                line=dict(
                    color=STAGE_COLORS.get(stage, "#8b949e"),
                    width=2.5,
                ),
                hovertemplate=f"Epoch {x[i]}: {stage}<extra></extra>",
            )
        )

    for stage, colour in STAGE_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=stage,
                line=dict(
                    color=colour,
                    width=2.5,
                ),
            )
        )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Predicted Hypnogram",
            font=dict(size=13, color="#8b949e"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(STAGE_Y.values()),
            ticktext=list(STAGE_Y.keys()),
            range=[0.3, 5.7],
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        xaxis=dict(
            title="Epoch",
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="#161b22",
            bordercolor="#21262d",
            borderwidth=1,
        ),
        height=280,
    )

    return fig


def _plot_confidence(
    uncertainty_df: pd.DataFrame,
    threshold: float,
) -> go.Figure:
    confidence = uncertainty_df["confidence"].tolist()
    entropy = uncertainty_df["entropy"].tolist()
    x = list(range(len(confidence)))

    colors = [
        "#f78166" if value <= threshold else "#7ee787"
        for value in confidence
    ]

    fig = go.Figure()

    fig.add_shape(
        type="line",
        x0=0,
        x1=max(len(confidence) - 1, 1),
        y0=threshold,
        y1=threshold,
        line=dict(
            color="#f78716",
            width=1.5,
            dash="dash",
        ),
    )

    fig.add_annotation(
        x=max(len(confidence) - 1, 0),
        y=min(threshold + 0.03, 0.98),
        text=f"threshold {threshold:.3f}",
        showarrow=False,
        font=dict(size=10, color="#f78716"),
        xanchor="right",
    )

    fig.add_trace(
        go.Bar(
            x=x,
            y=confidence,
            name="Confidence",
            marker=dict(
                color=colors,
                opacity=0.85,
            ),
            hovertemplate="Epoch %{x}<br>Confidence: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=entropy,
            name="Entropy",
            mode="lines",
            line=dict(
                color="#d2a8ff",
                width=1.5,
            ),
            yaxis="y2",
            opacity=0.7,
            hovertemplate="Epoch %{x}<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Confidence & Entropy per Epoch",
            font=dict(size=13, color="#8b949e"),
        ),
        yaxis=dict(
            title="Confidence",
            range=[0, 1.05],
            gridcolor="#21262d",
            linecolor="#30363d",
        ),
        yaxis2=dict(
            title="Entropy",
            overlaying="y",
            side="right",
            range=[0, 1.05],
            showgrid=False,
            linecolor="#30363d",
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#21262d",
            borderwidth=1,
        ),
        height=280,
    )

    return fig


def _plot_stage_distribution(stages: list[str]) -> go.Figure:
    from collections import Counter

    counts = Counter(stages)
    ordered = [stage for stage in STAGE_ORDER if stage in counts]
    values = [counts[stage] for stage in ordered]

    total = sum(values) or 1
    percentages = [value / total * 100 for value in values]

    fig = go.Figure(
        go.Bar(
            x=ordered,
            y=percentages,
            marker=dict(
                color=[STAGE_COLORS[stage] for stage in ordered],
                opacity=0.85,
                line=dict(color="#0d1117", width=1.5),
            ),
            text=[f"{percentage:.1f}%" for percentage in percentages],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
            hovertemplate="%{x}: %{y:.1f}% (%{customdata} epochs)<extra></extra>",
            customdata=values,
        )
    )

    y_max = (max(percentages) if percentages else 100) * 1.3

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Stage Distribution",
            font=dict(size=13, color="#8b949e"),
        ),
        yaxis=dict(
            title="% of recording",
            range=[0, y_max],
            gridcolor="#21262d",
        ),
        xaxis=dict(title="Sleep Stage"),
        height=240,
        showlegend=False,
    )

    return fig


def _plot_selective_curve(records) -> go.Figure:
    df_sel = pd.DataFrame(records)

    if df_sel.empty:
        return go.Figure()

    df_sel.columns = [str(column).lower() for column in df_sel.columns]

    coverage_col = next(
        (column for column in df_sel.columns if "coverage" in column),
        None,
    )
    accuracy_col = next(
        (
            column for column in df_sel.columns
            if "accuracy" in column or column == "acc"
        ),
        None,
    )

    if coverage_col is None or accuracy_col is None:
        return go.Figure()

    fig = go.Figure(
        go.Scatter(
            x=(df_sel[coverage_col] * 100).tolist(),
            y=(df_sel[accuracy_col] * 100).tolist(),
            mode="lines+markers",
            line=dict(
                color="#79c0ff",
                width=2.5,
            ),
            marker=dict(
                size=7,
                color="#58a6ff",
                line=dict(
                    color="#0d1117",
                    width=1.5,
                ),
            ),
            hovertemplate=(
                "Coverage: %{x:.0f}%<br>"
                "Accuracy: %{y:.1f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Selective Prediction Curve",
            font=dict(size=13, color="#8b949e"),
        ),
        xaxis=dict(
            title="Automatic coverage (%)",
            gridcolor="#21262d",
        ),
        yaxis=dict(
            title="Accuracy (%)",
            gridcolor="#21262d",
        ),
        height=260,
    )

    return fig


def _plot_confusion_matrix(cm_df: pd.DataFrame) -> go.Figure:
    # Extract stage names from index (true stages) and columns (predicted stages)
    # cm_df from build_confusion_matrix_df has index="true_{stage}" and columns="pred_{stage}"
    y_labels = [str(idx).replace("true_", "") for idx in cm_df.index]
    x_labels = [str(col).replace("pred_", "") for col in cm_df.columns]
    z = cm_df.values.tolist()

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=[[str(value) for value in row] for row in z],
            texttemplate="%{text}",
            colorscale=[[0, "#161b22"], [1, "#388bfd"]],
            showscale=False,
            hovertemplate=(
                "True: %{y}<br>"
                "Pred: %{x}<br>"
                "Count: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(
            text="Confusion Matrix",
            font=dict(size=13, color="#8b949e"),
        ),
        xaxis=dict(
            title="Predicted",
            gridcolor="rgba(0,0,0,0)",
            linecolor="#30363d",
        ),
        yaxis=dict(
            title="True",
            gridcolor="rgba(0,0,0,0)",
            linecolor="#30363d",
            autorange="reversed",
        ),
        height=300,
    )

    return fig


# ---------------------------------------------------------------------------
# Shared result renderer
# ---------------------------------------------------------------------------

def _render_results(result, queue, mode: str) -> None:
    stages = result.predicted_stages
    uncertainty_df = result.uncertainty_df
    has_labels = result.labels is not None

    n_total = queue.n_total
    n_review = queue.n_review
    n_retained = queue.n_retained
    threshold = queue.confidence_threshold
    mean_confidence = float(uncertainty_df["confidence"].mean())

    show_verified_labels = mode == "Research" and has_labels

    _section_header("Recording Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        _card(
            "Total Epochs",
            str(n_total),
            f"{n_total * 30 // 60} min",
            "#388bfd",
        ),
        unsafe_allow_html=True,
    )

    c2.markdown(
        _card(
            "Review Queue",
            str(n_review),
            f"{queue.review_fraction:.0%} of epochs",
            "#f78716",
        ),
        unsafe_allow_html=True,
    )

    c3.markdown(
        _card(
            "Auto-classified",
            str(n_retained),
            f"{1 - queue.review_fraction:.0%} retained",
            "#3fb950",
        ),
        unsafe_allow_html=True,
    )

    c4.markdown(
        _card(
            "Mean Confidence",
            f"{mean_confidence:.3f}",
            f"threshold {threshold:.3f}",
            "#a371f7",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)


    # ============================================================
    # HypnoTrust demo section: safe interpretation guide
    # Explanatory UI text only. No training/tuning executed.
    # ============================================================

    st.info(
        "HypnoTrust е proof-of-concept dashboard за анализ на sleep-stage predictions. "
        "Приложението показва не само какво предсказва моделът, а и къде е несигурен, "
        "кои епохи трябва да бъдат приоритизирани за човешки преглед и какви са "
        "най-честите failure modes. Не е предназначено за реална медицинска употреба."
    )

    with st.expander("How to interpret this dashboard", expanded=False):
        st.markdown(
            """
            **Main idea.** HypnoTrust използва вече замразени V5d predictions. 
            Този app не тренира модел и не променя резултатите.

            **Accuracy** показва дела правилно класифицирани епохи.

            **Mean confidence** показва средната увереност на модела. 
            По-ниска увереност често означава по-висок риск, но не гарантира грешка.

            **Hypnogram** сравнява true sleep-stage sequence с predicted sequence във времето.

            **Confidence plot** показва къде моделът е по-сигурен или по-несигурен по време на записа.

            **Review queue** съдържа най-ниско уверени епохи. 
            Целта не е моделът да замени специалист, а да насочи вниманието към най-рисковите части.

            **Review queue error rate** означава какъв процент от избраната queue са реални грешки. 
            Това не е процент от всички грешки в целия запис.

            **Retained accuracy** е accuracy върху епохите, които остават извън review queue.

            **Error capture fraction** показва каква част от всички грешки са попаднали в review queue.

            **Stage distribution** сравнява true и predicted брой епохи за Wake, N1, N2, N3 и REM. 
            Ако predicted N2 е прекалено висок, това може да показва N2-attractor behavior.

            **Error-pair chart** показва как точно греши моделът, например REM → N2 или N3 → N2.

            **Subject comparison** показва дали поведението на модела е стабилно между различни subjects.
            """
        )

    _section_header("Hypnogram", badge="PREDICTED")

    st.caption(
        "Hypnogram compares the reference sleep-stage sequence with the model prediction across time."
    )

    with st.expander("How to read the hypnogram", expanded=False):
        st.markdown(
            """
            Hypnogram показва sleep stages във времето. 
            Когато true и predicted линиите се разминават, моделът не следва правилно референтната последователност. 
            Продължителни разминавания са по-важни от единични грешки.
            """
        )

    st.plotly_chart(
        _plot_hypnogram(
            stages,
            true_stages=result.labels if show_verified_labels else None,
        ),
        use_container_width=True,
    )

    _section_header("Confidence & Entropy")

    st.plotly_chart(
        _plot_confidence(uncertainty_df, threshold),
        use_container_width=True,
    )

    st.markdown(
        f'<div style="font-size:0.77rem;color:#6e7681;margin-top:-0.4rem;">'
        f'Epochs below <b style="color:#f78716;">{threshold:.3f}</b> '
        f'are flagged for expert review '
        f'(lowest {queue.review_fraction:.0%} by confidence).</div>',
        unsafe_allow_html=True,
    )

    _section_header("Stage Distribution")

    st.plotly_chart(
        _plot_stage_distribution(stages),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Research":
        if has_labels and _METRICS_AVAILABLE:
            _section_header(
                "Classification Metrics",
                badge="VERIFIED LABELS",
                badge_color="#3fb950",
            )

            try:
                metrics = compute_metrics_if_available(
                    y_true=result.labels,
                    y_pred=result.predicted_stages,
                    probabilities=result.probabilities,
                    stage_names=result.stage_names,
                )
            except Exception as exc:
                st.error(f"Metrics computation failed: {exc}")
                metrics = None

            if metrics:
                accuracy_value = metrics.get("accuracy", np.nan)
                macro_f1_value = metrics.get("macro_f1", np.nan)
                kappa_value = metrics.get(
                    "cohen_kappa",
                    metrics.get("kappa", np.nan),
                )
                ece_value = metrics.get("ece", np.nan)

                m1, m2, m3, m4 = st.columns(4)

                m1.markdown(
                    _card(
                        "Accuracy",
                        f"{accuracy_value:.4f}",
                        "",
                        "#3fb950",
                    ),
                    unsafe_allow_html=True,
                )

                m2.markdown(
                    _card(
                        "Macro F1",
                        f"{macro_f1_value:.4f}",
                        "",
                        "#79c0ff",
                    ),
                    unsafe_allow_html=True,
                )

                m3.markdown(
                    _card(
                        "Kappa",
                        f"{kappa_value:.4f}",
                        "",
                        "#a371f7",
                    ),
                    unsafe_allow_html=True,
                )

                m4.markdown(
                    _card(
                        "ECE",
                        f"{ece_value:.4f}",
                        "",
                        "#f78716",
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                selective = metrics.get("selective_prediction")

                has_selective_curve = (
                    selective is not None
                    and hasattr(selective, "__len__")
                    and len(selective) > 0
                )

                if has_selective_curve:
                    _section_header("Selective Prediction")

                    st.plotly_chart(
                        _plot_selective_curve(selective),
                        use_container_width=True,
                    )

                col_cm, col_stage = st.columns(2)

                with col_cm:
                    _section_header("Confusion Matrix")

                    try:
                        cm_df = build_confusion_matrix_df(
                            result.labels,
                            result.predicted_stages,
                            result.stage_names,
                        )

                        st.plotly_chart(
                            _plot_confusion_matrix(cm_df),
                            use_container_width=True,
                        )
                    except Exception:
                        st.info("Confusion matrix unavailable.")

                with col_stage:
                    _section_header("Per-Stage Performance")

                    try:
                        per_stage_df = build_per_stage_metrics_df(
                            result.labels,
                            result.predicted_stages,
                            result.stage_names,
                        )

                        st.dataframe(
                            per_stage_df,
                            use_container_width=True,
                        )
                    except Exception:
                        st.info("Per-stage metrics unavailable.")

        elif has_labels and not _METRICS_AVAILABLE:
            st.warning(
                "Labels detected but `core.metrics` could not be imported. "
                "Metrics panel is disabled."
            )
        else:
            st.info(
                "📊 No verified **Stage** column found. "
                "Accuracy, F1, kappa, ECE, and confusion matrix require "
                "ground-truth labels."
            )

    _section_header(
        "Review Queue",
        badge=f"LOWEST {queue.review_fraction:.0%}",
    )

    display_cols = [
        "epoch",
        "predicted_stage",
        "confidence",
        "entropy",
    ]

    if mode == "Research" and "true_stage" in queue.review_table.columns:
        display_cols = [
            "epoch",
            "true_stage",
            "predicted_stage",
            "confidence",
            "entropy",
        ]

    review_display = queue.review_table[
        [col for col in display_cols if col in queue.review_table.columns]
    ].copy()

    for col in ("confidence", "entropy"):
        if col in review_display.columns:
            review_display[col] = review_display[col].map("{:.4f}".format)

    st.dataframe(
        review_display.rename(
            columns={
                "epoch": "Epoch",
                "predicted_stage": "Predicted",
                "true_stage": "True Stage",
                "confidence": "Confidence",
                "entropy": "Entropy",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📋 Full prediction table", expanded=False):
        full_display = uncertainty_df.copy()

        if "predicted_stage" in full_display.columns:
            full_display["predicted_stage"] = stages
        else:
            insert_at = (
                full_display.columns.get_loc("confidence")
                if "confidence" in full_display.columns
                else 0
            )
            full_display.insert(insert_at, "predicted_stage", stages)

        if show_verified_labels:
            full_display["true_stage"] = result.labels

        cols_to_show = [
            col
            for col in [
                "epoch",
                "true_stage",
                "predicted_stage",
                "confidence",
                "entropy",
            ]
            if col in full_display.columns
        ]

        st.dataframe(
            full_display[cols_to_show],
            use_container_width=True,
            hide_index=True,
        )

    _section_header("Export")

    ex1, ex2, _ = st.columns([1, 1, 2])

    try:
        pred_bytes = predictions_to_csv_bytes(result)
    except Exception:
        pred_bytes = uncertainty_df.to_csv(index=False).encode("utf-8")

    try:
        queue_bytes = review_queue_to_csv_bytes(queue)
    except Exception:
        queue_bytes = queue.review_table.to_csv(index=False).encode("utf-8")

    ex1.download_button(
        "⬇ Predictions CSV",
        data=pred_bytes,
        file_name="hypnotrust_predictions.csv",
        mime="text/csv",
    )

    ex2.download_button(
        "⬇ Review Queue CSV",
        data=queue_bytes,
        file_name="hypnotrust_review_queue.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> tuple[str, dict]:
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:0.8rem 0 0.3rem;">'
            '<span style="font-size:1.6rem;">🔬</span>'
            '<div style="font-size:1.1rem;font-weight:700;color:#e6edf3;'
            'margin-top:0.2rem;">HypnoTrust</div>'
            '<div style="font-size:0.7rem;color:#6e7681;letter-spacing:0.06em;">'
            'RESEARCH PROTOTYPE v0.1</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        mode = st.radio(
            "Mode",
            options=["User", "Demo", "EDF Upload", "Research", "Transition Explorer"],
            index=1,
            help=(
                "User: upload feature CSV, predictions only.\n"
                "Demo: synthetic profiles through the real model.\n"
                "Research: full metrics when verified labels are present.\n"
                "Transition Explorer: analyze N2↔N3 transitions in real EEG."
            ),
        )

        _mode_badge(mode)

        st.markdown("<hr>", unsafe_allow_html=True)

        params: dict = {}

        if mode == "Demo":
            params["profile"] = st.selectbox(
                "Sleep profile",
                list(_DEMO_PROFILES.keys()),
            )

            params["n_epochs"] = st.slider(
                "Recording length (epochs)",
                40,
                200,
                80,
                10,
            )

            params["seed"] = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    step=1,
                    help="Change to generate a different realisation.",
                )
            )

        params["review_pct"] = st.slider(
            "Review queue",
            10,
            50,
            20,
            5,
            format="%d%%",
            help="Percentage of lowest-confidence epochs sent for human review.",
        )

        params["review_fraction"] = params["review_pct"] / 100

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.7rem;color:#6e7681;line-height:1.7;">'
            f'<b style="color:#8b949e;">Model:</b> Temporal v2 RF<br>'
            f'<b style="color:#8b949e;">Policy:</b> Uncertainty v3<br>'
            f'<b style="color:#8b949e;">Stages:</b> Wake · N1 · N2 · N3 · REM<br>'
            f'<b style="color:#8b949e;">Epoch:</b> 30 s<br>'
            f'<b style="color:#8b949e;">Features:</b> '
            f'{len(_TEMPORAL_FEATURE_COLUMNS)} temporal</div>',
            unsafe_allow_html=True,
        )

    return mode, params


def _run_transition_explorer(params: dict) -> None:
    """Run Transition Explorer mode for EEG analysis."""
    _section_header(
        "Transition Explorer",
        badge="EEG SPECTRAL ANALYSIS",
        badge_color="#d2a8ff",
    )

    st.markdown(
        '<div style="color:#8b949e;font-size:0.81rem;margin-bottom:1rem;">'
        'Analyze N2↔N3 transitions in real Sleep-EDF recordings. '
        'This is exploratory research, not clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ EXPLORATORY RESEARCH MODE ONLY. "
        "Do not use for clinical diagnosis. "
        "This is a research prototype for studying EEG spectral changes at sleep-stage transitions."
    )

    psg_file = st.file_uploader(
        "Upload PSG EDF",
        type=["edf"],
        key="transition_psg_upload",
    )

    hypnogram_file = st.file_uploader(
        "Upload Hypnogram EDF",
        type=["edf"],
        key="transition_hypnogram_upload",
    )

    if psg_file is None:
        st.info("Upload a PSG EDF file and its corresponding hypnogram to begin.")
        return

    upload_signature = f"{psg_file.name}:{psg_file.size}"

    # Transition direction selection
    transition_kind = st.selectbox(
        "Transition Direction",
        options=["N2->N3", "N3->N2", "BOTH"],
        index=2,
    )

    # Stability criterion
    k_value = st.selectbox(
        "Stability Criterion (k)",
        options=[1, 2, 3, 5],
        index=2,
        help="Number of consecutive epochs required before and after the boundary."
    )

    if st.button("▶  Detect Transitions", type="primary"):
        psg_path = None
        hypnogram_path = None

        try:
            import mne

            psg_path = _save_uploaded_edf(psg_file, ".edf")

            with st.spinner("Loading PSG and hypnogram…"):
                raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")

            # Load hypnogram
            if hypnogram_file is not None:
                hypnogram_path = _save_uploaded_edf(hypnogram_file, ".edf")
                annotations = mne.read_annotations(hypnogram_path)
                st.success(f"Hypnogram loaded: {len(annotations)} segments.")
            else:
                st.error("Hypnogram EDF is required for transition detection.")
                return

            # Extract labels from annotations
            from core.feature_extraction import _labels_from_annotations

            n_epochs = int(raw.n_times / raw.info["sfreq"] // 30)
            labels = _labels_from_annotations(annotations, n_epochs, _CFG.epoch_length_sec)

            if labels is None:
                st.error("Could not extract valid labels from hypnogram.")
                return

            # Detect transitions
            with st.spinner("Detecting transitions…"):
                events = detect_transitions(labels, _CFG.epoch_length_sec, k=k_value)

            # Filter by kind
            filtered_events = filter_transitions_by_kind(events, transition_kind)

            st.session_state["transition_events"] = filtered_events
            st.session_state["transition_raw"] = raw
            st.session_state["transition_labels"] = labels
            st.session_state["transition_upload_signature"] = upload_signature
            st.session_state["transition_kind"] = transition_kind
            st.session_state["transition_k"] = k_value

        except Exception as exc:
            st.error(f"Transition detection error: {exc}")
            import traceback
            st.error(traceback.format_exc())

        finally:
            for tmp_path in (psg_path, hypnogram_path):
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    # Display results
    if (
        "transition_events" in st.session_state
        and st.session_state.get("transition_upload_signature") == upload_signature
    ):
        events = st.session_state["transition_events"]
        raw = st.session_state["transition_raw"]
        labels = st.session_state["transition_labels"]
        transition_kind = st.session_state["transition_kind"]
        k_value = st.session_state["transition_k"]

        _section_header("Transition Detection Results")

        counts = {
            "N2->N3": sum(1 for e in events if e.kind == "N2->N3"),
            "N3->N2": sum(1 for e in events if e.kind == "N3->N2"),
            "BOTH": len(events),
        }

        c1, c2, c3 = st.columns(3)

        c1.markdown(
            _card(
                f"{transition_kind} Transitions",
                str(counts[transition_kind]),
                f"k={k_value} criterion",
                "#d2a8ff",
            ),
            unsafe_allow_html=True,
        )

        c2.markdown(
            _card(
                "Total Detected",
                str(counts["BOTH"]),
                f"Across recording",
                "#a371f7",
            ),
            unsafe_allow_html=True,
        )

        c3.markdown(
            _card(
                "N2→N3",
                str(counts["N2->N3"]),
                f"N3→N2: {counts['N3->N2']}",
                "#7ee787",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if not events:
            st.info(f"No {transition_kind} transitions detected with k={k_value} criterion.")
            return

        # Event selection
        _section_header("Select Transition Event")

        event_options = [
            f"Epoch {e.epoch_index} ({e.kind}) at {e.boundary_seconds:.0f}s"
            for e in events
        ]

        selected_idx = st.selectbox(
            "Select event to analyze",
            range(len(event_options)),
            format_func=lambda x: event_options[x],
        )

        selected_event = events[selected_idx]

        # Display selected event details
        st.info(
            f"Selected: Epoch {selected_event.epoch_index}, {selected_event.kind} at {selected_event.boundary_seconds:.0f}s "
            f"({selected_event.before_stage} → {selected_event.after_stage})"
        )

        # Spectral analysis button
        if st.button("▶  Analyze EEG Spectral Power", type="primary"):
            with st.spinner("Computing spectral analysis…"):
                event_boundary_sample = int(selected_event.boundary_seconds * raw.info["sfreq"])

                # Analyze Fpz-Cz
                try:
                    fpz_segment = extract_eeg_segment(
                        raw,
                        "Fpz-Cz",
                        event_boundary_sample,
                        (-60.0, 60.0),
                    )

                    fpz_windows = compute_spectral_windows(
                        fpz_segment.signal_uv,
                        fpz_segment.sfreq,
                        event_boundary_sample,
                    )

                    fpz_before_after = compute_before_after_power(fpz_windows)

                    st.session_state["fpz_windows"] = fpz_windows
                    st.session_state["fpz_before_after"] = fpz_before_after
                except Exception as exc:
                    st.error(f"Fpz-Cz analysis error: {exc}")

                # Analyze Pz-Oz
                try:
                    pz_segment = extract_eeg_segment(
                        raw,
                        "Pz-Oz",
                        event_boundary_sample,
                        (-60.0, 60.0),
                    )

                    pz_windows = compute_spectral_windows(
                        pz_segment.signal_uv,
                        pz_segment.sfreq,
                        event_boundary_sample,
                    )

                    pz_before_after = compute_before_after_power(pz_windows)

                    st.session_state["pz_windows"] = pz_windows
                    st.session_state["pz_before_after"] = pz_before_after
                except Exception as exc:
                    st.error(f"Pz-Oz analysis error: {exc}")

        # Display spectral results
        if "fpz_windows" in st.session_state:
            _section_header("Spectral Power Analysis (0.5-2 Hz)")

            fpz_before_after = st.session_state["fpz_before_after"]
            pz_before_after = st.session_state["pz_before_after"]

            s1, s2 = st.columns(2)

            s1.markdown(
                _card(
                    "Fpz-Cz Before",
                    f"{fpz_before_after['before_power_db']:.2f} dB",
                    f"After: {fpz_before_after['after_power_db']:.2f} dB",
                    "#79c0ff",
                ),
                unsafe_allow_html=True,
            )

            s2.markdown(
                _card(
                    "Pz-Oz Before",
                    f"{pz_before_after['before_power_db']:.2f} dB",
                    f"After: {pz_before_after['after_power_db']:.2f} dB",
                    "#7ee787",
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"**Power Change:** Fpz-Cz: {fpz_before_after['power_change_db']:+.2f} dB, "
                f"Pz-Oz: {pz_before_after['power_change_db']:+.2f} dB"
            )

            # Plot spectral curves
            _section_header("Spectral Power Curves")

            fpz_windows = st.session_state["fpz_windows"]
            pz_windows = st.session_state["pz_windows"]

            fpz_times = [w.relative_center_s for w in fpz_windows]
            fpz_power_db = [w.power_db_uv2 for w in fpz_windows]

            pz_times = [w.relative_center_s for w in pz_windows]
            pz_power_db = [w.power_db_uv2 for w in pz_windows]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=fpz_times,
                    y=fpz_power_db,
                    mode="lines",
                    name="Fpz-Cz",
                    line=dict(color="#79c0ff", width=2.5),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pz_times,
                    y=pz_power_db,
                    mode="lines",
                    name="Pz-Oz",
                    line=dict(color="#7ee787", width=2.5),
                )
            )

            fig.add_vline(x=0, line_dash="dash", line_color="#f78716", annotation_text="Boundary")

            fig.update_layout(
                **PLOTLY_BASE,
                title=dict(
                    text="Slow-Wave (0.5-2 Hz) Power Around Transition",
                    font=dict(size=13, color="#8b949e"),
                ),
                xaxis=dict(
                    title="Time relative to boundary (s)",
                    gridcolor="#21262d",
                    linecolor="#30363d",
                ),
                yaxis=dict(
                    title="Power (dB re 1µV²)",
                    gridcolor="#21262d",
                    linecolor="#30363d",
                ),
                legend=dict(
                    bgcolor="#161b22",
                    bordercolor="#21262d",
                    borderwidth=1,
                ),
                height=300,
            )

            st.plotly_chart(fig, use_container_width=True)

        # Disclaimer
        st.divider()
        st.warning(
            "**Methodological Notes:** "
            f"k={k_value} stability criterion was selected exploratorily after examining E0 annotation counts, not preregistered. "
            "Stage boundaries are 30-s label boundaries, not precise physiological transition moments. "
            "SC4001 is a development/training-exposed recording. "
            "No population inference is justified from a single participant."
        )




# ---------------------------------------------------------------------------
# EDF Upload mode
# ---------------------------------------------------------------------------

def _render_edf_evaluation_dashboard(evaluation_result) -> None:
    """
    Render the EDF evaluation dashboard with metrics, confusion matrix, and per-stage performance.
    """
    st.divider()

    _section_header(
        "EDF Evaluation Dashboard",
        badge="SCIENTIFIC METRICS",
        badge_color="#3fb950",
    )

    provenance = evaluation_result.provenance

    # Scientific provenance banner
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;">'
        f'<div style="color:#8b949e;font-size:0.76rem;margin-bottom:0.4rem;">'
        f'<b style="color:#e6edf3;">Recording:</b> {provenance["recording_identity"]} · '
        f'<b style="color:#e6edf3;">Model:</b> {provenance["model_name"]} · '
        f'<b style="color:#e6edf3;">Uncertainty Policy:</b> {provenance["uncertainty_policy"]} · '
        f'<b style="color:#e6edf3;">Epoch:</b> {provenance["epoch_length_sec"]}s</div>'
        f'<div style="color:#6e7681;font-size:0.74rem;">'
        f'Total epochs: {provenance["total_epochs"]} · '
        f'Valid labels: {provenance["valid_epochs"]} · '
        f'Excluded: {provenance["excluded_epochs"]} · '
        f'Evaluated: {provenance["evaluation_timestamp"]}</div></div>',
        unsafe_allow_html=True,
    )

    # Development exposure warning for SC4001
    if "SC4001" in provenance.get("recording_identity", ""):
        st.warning(
            "⚠️ **Development/Training-Exposed Recording**: "
            "SC4001 is part of the Sleep-EDF training set. "
            "The 97%+ accuracy reflects development performance, not external validation. "
            "Do not cite this as generalization evidence."
        )

    # Epoch counts
    c1, c2, c3 = st.columns(3)

    c1.markdown(
        _card(
            "Total Epochs",
            str(evaluation_result.n_total_epochs),
            f"{evaluation_result.n_total_epochs * 30 // 60} min",
            "#388bfd",
        ),
        unsafe_allow_html=True,
    )

    c2.markdown(
        _card(
            "Valid Labels",
            str(evaluation_result.n_valid_epochs),
            f"{evaluation_result.n_valid_epochs / evaluation_result.n_total_epochs:.1%} coverage",
            "#3fb950",
        ),
        unsafe_allow_html=True,
    )

    c3.markdown(
        _card(
            "Excluded",
            str(evaluation_result.n_excluded_epochs),
            "Invalid annotations",
            "#f78716",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # If no valid labels, show message and return
    if evaluation_result.n_valid_epochs == 0:
        st.info(
            "No valid hypnogram labels found. "
            "Evaluation metrics require verified stage annotations. "
            "Showing predictions only."
        )
        return

    # Synchronized hypnogram
    _section_header("Synchronized Hypnogram")

    result = st.session_state.get("edf_result")
    if result and result.labels:
        st.plotly_chart(
            _plot_synchronized_hypnogram(
                true_stages=result.labels,
                predicted_stages=result.predicted_stages,
                confidences=result.uncertainty_df["confidence"].values,
                highlight_errors=True,
            ),
            use_container_width=True,
        )

        st.caption(
            "Green segments: correct predictions. Red segments: prediction errors. "
            "Dotted line: verified hypnogram labels."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main metrics
    metrics = evaluation_result.metrics

    m1, m2, m3, m4 = st.columns(4)

    m1.markdown(
        _card(
            "Accuracy",
            f"{metrics['accuracy']:.4f}",
            "",
            "#3fb950",
        ),
        unsafe_allow_html=True,
    )

    m2.markdown(
        _card(
            "Balanced Accuracy",
            f"{metrics['balanced_accuracy']:.4f}",
            "",
            "#79c0ff",
        ),
        unsafe_allow_html=True,
    )

    m3.markdown(
        _card(
            "Macro F1",
            f"{metrics['macro_f1']:.4f}",
            "",
            "#a371f7",
        ),
        unsafe_allow_html=True,
    )

    m4.markdown(
        _card(
            "Cohen's Kappa",
            f"{metrics['cohen_kappa']:.4f}",
            "",
            "#f78716",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Wake metrics and sleep-only metrics
    if evaluation_result.wake_only_metrics and evaluation_result.sleep_only_metrics:
        _section_header("Subset Metrics")

        wake_recall = evaluation_result.wake_only_metrics["wake_recall"]
        wake_n = evaluation_result.wake_only_metrics["wake_n_epochs"]

        sleep_acc = evaluation_result.sleep_only_metrics["sleep_only_accuracy"]
        sleep_f1 = evaluation_result.sleep_only_metrics["sleep_only_macro_f1"]
        sleep_n = evaluation_result.sleep_only_metrics["sleep_only_n_epochs"]

        s1, s2, s3 = st.columns(3)

        s1.markdown(
            _card(
                "Wake Recall",
                f"{wake_recall:.4f}",
                f"{wake_n} epochs",
                "#58a6ff",
            ),
            unsafe_allow_html=True,
        )

        s2.markdown(
            _card(
                "Sleep-Only Accuracy",
                f"{sleep_acc:.4f}",
                f"F1: {sleep_f1:.4f} · {sleep_n} epochs",
                "#7ee787",
            ),
            unsafe_allow_html=True,
        )

        # Always-Wake Baseline
        if evaluation_result.always_wake_baseline:
            baseline_acc = evaluation_result.always_wake_baseline["always_wake_baseline_accuracy"]
            total_n = evaluation_result.always_wake_baseline["total_epochs"]

            s3.markdown(
                _card(
                    "Always-Wake Baseline",
                    f"{baseline_acc:.4f}",
                    f"{total_n} epochs",
                    "#6e7681",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrix
    _section_header("Confusion Matrix")

    cm_df = evaluation_result.confusion_matrix

    st.plotly_chart(
        _plot_confusion_matrix(cm_df),
        use_container_width=True,
    )

    # Per-stage metrics
    _section_header("Per-Stage Performance")

    per_stage_df = evaluation_result.per_stage_metrics

    st.dataframe(
        per_stage_df,
        use_container_width=True,
        hide_index=True,
    )

    # Export section
    _section_header("Export Evaluation Results")

    ex1, ex2, _ = st.columns([1, 1, 2])

    # Prepare CSV export
    result = st.session_state.get("edf_result")
    if result:
        csv_bytes = export_evaluation_csv(
            y_true=result.labels,
            y_pred=result.predicted_stages,
            probabilities=result.probabilities,
            confidences=result.uncertainty_df["confidence"].values,
            epoch_length_sec=_CFG.epoch_length_sec,
            recording_identity=provenance["recording_identity"],
        )

        ex1.download_button(
            "⬇ Evaluation CSV",
            data=csv_bytes,
            file_name=f"hypnotrust_edf_evaluation_{provenance['recording_identity']}.csv",
            mime="text/csv",
        )

    # Prepare JSON export
    json_bytes = export_evaluation_json(
        evaluation_result=evaluation_result,
        recording_identity=provenance["recording_identity"],
    )

    ex2.download_button(
        "⬇ Evaluation JSON",
        data=json_bytes,
        file_name=f"hypnotrust_edf_evaluation_{provenance['recording_identity']}.json",
        mime="application/json",
    )


def _save_uploaded_edf(uploaded_file, suffix=".edf") -> str:
    """Save a Streamlit UploadedFile to a temporary EDF path for MNE."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _run_edf_upload(params: dict) -> None:
    _section_header(
        "EDF Upload",
        badge="EXPERIMENTAL",
        badge_color="#f78716",
    )

    st.markdown(
        '<div style="color:#8b949e;font-size:0.81rem;margin-bottom:1rem;">'
        'Upload a PSG EDF file and run it through the real HypnoTrust pipeline: '
        'MNE EDF reader → feature extraction → Temporal v2 model → uncertainty → review queue. '
        '<b style="color:#f78716;">Experimental research mode only.</b></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Use this only with compatible PSG EDF files containing the expected Sleep-EDF style channels "
        "such as Fpz-Cz, Pz-Oz and horizontal EOG. Hypnogram/annotation EDF is optional.",
        icon="ℹ️",
    )

    psg_file = st.file_uploader(
        "Upload PSG EDF",
        type=["edf"],
        key="edf_psg_upload",
    )

    hypnogram_file = st.file_uploader(
        "Optional: upload hypnogram / annotations EDF",
        type=["edf"],
        key="edf_hypnogram_upload",
    )

    if psg_file is None:
        return

    upload_signature = f"{psg_file.name}:{psg_file.size}"

    if st.button("▶  Run EDF Pipeline", type="primary"):
        psg_path = None
        hypnogram_path = None

        try:
            import mne

            psg_path = _save_uploaded_edf(psg_file, ".edf")

            with st.spinner("Reading PSG EDF with MNE…"):
                raw = mne.io.read_raw_edf(
                    psg_path,
                    preload=False,
                    verbose="ERROR",
                )

            annotations = None

            if hypnogram_file is not None:
                hypnogram_path = _save_uploaded_edf(hypnogram_file, ".edf")

                try:
                    annotations = mne.read_annotations(hypnogram_path)
                    st.success(f"Annotations loaded: {len(annotations)} segments.")
                except Exception as exc:
                    st.warning(
                        f"Could not read hypnogram annotations: {exc}. "
                        "Continuing with PSG only."
                    )
                    annotations = None

            with st.spinner("Running full EDF pipeline…"):
                result = run_full_pipeline(
                    raw,
                    annotations=annotations,
                    config=_CFG,
                    subject=0,
                    recording=0,
                )

            queue = build_review_queue(
                uncertainty_df=result.uncertainty_df,
                predicted_stages=result.predicted_stages,
                fraction=params["review_fraction"],
                labels=None,
            )

            # Run evaluation if annotations are available
            evaluation_result = None
            if result.labels is not None:
                with st.spinner("Computing evaluation metrics…"):
                    raw_duration_sec = float(raw.n_times / raw.info["sfreq"])

                    # Validate annotation alignment
                    if annotations is not None:
                        annotation_tuples = [
                            (onset, duration, desc)
                            for onset, duration, desc in zip(
                                annotations.onset,
                                annotations.duration,
                                annotations.description,
                            )
                        ]
                        is_aligned, alignment_warning = validate_annotation_alignment(
                            raw_duration_sec=raw_duration_sec,
                            annotations=annotation_tuples,
                            epoch_length_sec=_CFG.epoch_length_sec,
                        )

                        if not is_aligned:
                            st.warning(
                                f"⚠️ Annotation alignment warning: {alignment_warning} "
                                "Evaluation metrics may be unreliable."
                            )

                    evaluation_result = evaluate_edf_predictions(
                        y_true=result.labels,
                        y_pred=result.predicted_stages,
                        probabilities=result.probabilities,
                        stage_names=result.stage_names,
                        raw_duration_sec=raw_duration_sec,
                        annotations_provided=annotations is not None,
                        model_name="Temporal v2 RandomForest",
                        uncertainty_policy=_CFG.uncertainty_policy,
                        epoch_length_sec=_CFG.epoch_length_sec,
                        recording_identity=psg_file.name,
                    )

            st.session_state["edf_result"] = result
            st.session_state["edf_queue"] = queue
            st.session_state["edf_evaluation"] = evaluation_result
            st.session_state["edf_upload_signature"] = upload_signature

        except Exception as exc:
            st.error(f"EDF pipeline error: {exc}")

        finally:
            for tmp_path in (psg_path, hypnogram_path):
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    if (
        "edf_result" in st.session_state
        and st.session_state.get("edf_upload_signature") == upload_signature
    ):
        _render_results(
            st.session_state["edf_result"],
            st.session_state["edf_queue"],
            mode="EDF Upload",
        )

        # Render evaluation dashboard if annotations were provided
        evaluation_result = st.session_state.get("edf_evaluation")
        if evaluation_result is not None:
            _render_edf_evaluation_dashboard(evaluation_result)


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _run_demo(params: dict) -> None:
    _section_header(
        "Demo Mode",
        badge="SYNTHETIC PROFILE",
        badge_color="#388bfd",
    )

    st.markdown(
        '<div style="color:#8b949e;font-size:0.81rem;margin-bottom:1rem;">'
        'A synthetic sleep recording is generated from a physiological '
        'template and run through the '
        '<b style="color:#e6edf3;">real Temporal v2 model</b>. '
        'Configure the profile and seed in the sidebar, '
        'then click <b style="color:#e6edf3;">Generate</b>.</div>',
        unsafe_allow_html=True,
    )

    if st.button("▶  Generate Demo", type="primary"):
        feature_df, _template_labels = _make_demo_feature_df(
            profile_name=params["profile"],
            n_epochs=params["n_epochs"],
            seed=params["seed"],
        )

        # Demo labels are not passed as verified labels.
        # Demo is for workflow/UI only, not for reporting metrics.
        result, queue = _run_pipeline(
            feature_df=feature_df,
            labels=None,
            review_fraction=params["review_fraction"],
            mode="Demo",
        )

        if result is not None and queue is not None:
            st.session_state["demo_result"] = result
            st.session_state["demo_queue"] = queue
            st.session_state["demo_meta"] = {
                "profile": params["profile"],
                "n_epochs": params["n_epochs"],
                "seed": params["seed"],
            }

    if "demo_result" in st.session_state:
        meta = st.session_state["demo_meta"]

        st.markdown(
            f'<div style="font-size:0.77rem;color:#6e7681;'
            f'margin-bottom:0.8rem;">'
            f'Profile: <b style="color:#79c0ff;">{meta["profile"]}</b> · '
            f'{meta["n_epochs"]} epochs · seed {meta["seed"]}</div>',
            unsafe_allow_html=True,
        )

        _render_results(
            st.session_state["demo_result"],
            st.session_state["demo_queue"],
            mode="Demo",
        )
    else:
        st.markdown(
            '<div style="background:#161b22;border:1px solid #21262d;'
            'border-radius:12px;padding:2.5rem;text-align:center;'
            'color:#6e7681;font-size:0.9rem;">'
            'Click <b style="color:#e6edf3;">Generate Demo</b> to run '
            'the pipeline on a synthetic recording.</div>',
            unsafe_allow_html=True,
        )


def _run_user(params: dict) -> None:
    _section_header(
        "User Mode",
        badge="PREDICTIONS ONLY",
        badge_color="#3fb950",
    )

    n_features = len(_TEMPORAL_FEATURE_COLUMNS)

    st.markdown(
        f'<div style="color:#8b949e;font-size:0.81rem;margin-bottom:1rem;">'
        f'Upload a feature CSV. Accepted formats:<br>'
        f'&nbsp;• <b style="color:#e6edf3;">Temporal v2 CSV</b>: '
        f'{n_features} feature columns '
        f'(e.g. <code>{_TEMPORAL_FEATURE_COLUMNS[0]}</code> … '
        f'<code>{_TEMPORAL_FEATURE_COLUMNS[-1]}</code>)<br>'
        f'&nbsp;• <b style="color:#e6edf3;">Baseline CSV</b>: '
        f'13 feature columns + Subject, Recording, Epoch '
        f'(Stage optional; auto-expanded to {n_features} temporal features)<br>'
        f'Accuracy and evaluation metrics are not shown in this mode.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload feature CSV",
        type=["csv"],
        key="user_csv",
    )

    if uploaded is None:
        return

    upload_signature = f"{uploaded.name}:{uploaded.size}"

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    feature_df, _labels, error, schema_name = _resolve_feature_df(raw_df)

    if error:
        st.error(error)
        return

    st.success(
        f"✓ CSV schema detected: `{schema_name}`. "
        f"Rows: {len(feature_df)}. "
        f"Features: {len(_TEMPORAL_FEATURE_COLUMNS)}."
    )

    if st.button("▶  Run Prediction", type="primary"):
        result, queue = _run_pipeline(
            feature_df=feature_df,
            labels=None,
            review_fraction=params["review_fraction"],
            mode="User",
        )

        if result is not None and queue is not None:
            st.session_state["user_result"] = result
            st.session_state["user_queue"] = queue
            st.session_state["user_upload_signature"] = upload_signature

    if (
        "user_result" in st.session_state
        and st.session_state.get("user_upload_signature") == upload_signature
    ):
        _render_results(
            st.session_state["user_result"],
            st.session_state["user_queue"],
            mode="User",
        )


def _run_research(params: dict) -> None:
    _section_header(
        "Research Mode",
        badge="FULL ANALYSIS",
        badge_color="#a371f7",
    )

    st.markdown(
        '<div style="color:#8b949e;font-size:0.81rem;margin-bottom:1rem;">'
        'Upload a feature CSV in temporal v2 or baseline format. '
        'If a <b style="color:#e6edf3;">Stage</b> column with verified '
        'ground-truth labels is present, classification metrics, '
        'confusion matrix, and selective prediction are unlocked.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload feature CSV",
        type=["csv"],
        key="research_csv",
    )

    if uploaded is None:
        return

    upload_signature = f"{uploaded.name}:{uploaded.size}"

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    feature_df, labels, error, schema_name = _resolve_feature_df(raw_df)

    if error:
        st.error(error)
        return

    st.success(
        f"✓ CSV schema detected: `{schema_name}`. "
        f"Rows: {len(feature_df)}. "
        f"Features: {len(_TEMPORAL_FEATURE_COLUMNS)}."
    )

    if labels is not None:
        st.success(
            f"✓ Verified Stage labels detected "
            f"({len(set(labels))} classes, {len(labels)} epochs). "
            f"Metrics unlocked."
        )
    else:
        st.warning(
            "No verified **Stage** column found. "
            "Showing predictions and uncertainty only."
        )

    if st.button("▶  Run Analysis", type="primary"):
        result, queue = _run_pipeline(
            feature_df=feature_df,
            labels=labels,
            review_fraction=params["review_fraction"],
            mode="Research",
        )

        if result is not None and queue is not None:
            st.session_state["research_result"] = result
            st.session_state["research_queue"] = queue
            st.session_state["research_upload_signature"] = upload_signature

    if (
        "research_result" in st.session_state
        and st.session_state.get("research_upload_signature") == upload_signature
    ):
        _render_results(
            st.session_state["research_result"],
            st.session_state["research_queue"],
            mode="Research",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(
        '<div style="padding:1.1rem 0 0.3rem;">'
        '<div style="display:flex;align-items:center;gap:0.8rem;">'
        '<span style="font-size:2rem;">🔬</span>'
        '<div>'
        '<div style="font-size:1.45rem;font-weight:700;color:#e6edf3;'
        'line-height:1.1;">HypnoTrust</div>'
        '<div style="font-size:0.76rem;color:#6e7681;letter-spacing:0.04em;">'
        'Uncertainty-Aware Sleep Stage Classification · Research Prototype'
        '</div></div></div></div>',
        unsafe_allow_html=True,
    )

    _disclaimer()

    mode, params = _render_sidebar()

    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Demo":
        _run_demo(params)
    elif mode == "User":
        _run_user(params)
    elif mode == "EDF Upload":
        _run_edf_upload(params)
    elif mode == "Research":
        _run_research(params)
    elif mode == "Transition Explorer":
        _run_transition_explorer(params)

    st.divider()


    # ============================================================

    # ============================================================
    # HypnoTrust demo section: safe subject comparison dashboard
    # Self-contained comparison from frozen V5d predictions.
    # No training/tuning executed.
    # ============================================================

    st.divider()

    st.subheader("Subject comparison dashboard")

    st.caption(
        "Compares model behavior across subjects: epochs, accuracy, confidence, error rate and dominant error type."
    )

    with st.expander("How to read subject comparison", expanded=False):
        st.markdown(
            """
            Тази таблица показва дали моделът се държи стабилно при различни subjects. 
            Големи разлики в accuracy или confidence са важни, защото sleep EEG има силна индивидуална вариабилност. 
            `Top error pair` показва най-честия тип грешка за даден subject.
            """
        )

    from pathlib import Path as _HypnoTrustComparisonPath
    import pandas as _hypnotrust_comparison_pd

    # Fixed: Use project-relative path instead of hard-coded Windows path
    _hypnotrust_comparison_predictions_path = Path(__file__).parent.parent / "experiments" / "v5_expanded_cross_subject_evaluation" / "v5d_frozen_external_evaluation" / "v5d_epoch_predictions.csv"

    if not _hypnotrust_comparison_predictions_path.exists():
        st.warning("Subject comparison unavailable: predictions CSV not found.")
    else:
        _comparison_df = _hypnotrust_comparison_pd.read_csv(
            _hypnotrust_comparison_predictions_path
        )

        _comparison_required_columns = {
            "subject_code",
            "true_stage",
            "pred_stage",
            "confidence",
            "correct",
        }
        _comparison_missing_columns = _comparison_required_columns.difference(
            _comparison_df.columns
        )

        if _comparison_missing_columns:
            st.warning(
                "Subject comparison unavailable: missing columns "
                + ", ".join(sorted(_comparison_missing_columns))
            )
        else:
            _comparison_role_filter = locals().get("selected_role")

            if (
                _comparison_role_filter is not None
                and "v5_role" in _comparison_df.columns
                and str(_comparison_role_filter)
                in _comparison_df["v5_role"].astype(str).unique().tolist()
            ):
                _comparison_df = _comparison_df[
                    _comparison_df["v5_role"].astype(str)
                    == str(_comparison_role_filter)
                ].copy()

            if _comparison_df.empty:
                st.info("No subject comparison data available.")
            else:
                _comparison_correct_series = _comparison_df["correct"]

                if str(_comparison_correct_series.dtype) == "bool":
                    _comparison_df["_is_correct_for_comparison"] = (
                        _comparison_correct_series
                    )
                else:
                    _comparison_df["_is_correct_for_comparison"] = (
                        _comparison_correct_series
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1", "yes"])
                    )

                _subject_summary_table = (
                    _comparison_df
                    .groupby("subject_code")
                    .agg(
                        epochs=("subject_code", "size"),
                        accuracy=("_is_correct_for_comparison", "mean"),
                        mean_confidence=("confidence", "mean"),
                    )
                    .reset_index()
                    .sort_values("subject_code")
                )

                _subject_summary_table["error_rate"] = (
                    1.0 - _subject_summary_table["accuracy"]
                )

                if "v5_role" in _comparison_df.columns:
                    _subject_roles = (
                        _comparison_df
                        .groupby("subject_code")["v5_role"]
                        .agg(lambda values: str(values.iloc[0]))
                        .reset_index()
                        .rename(columns={"v5_role": "role"})
                    )
                    _subject_summary_table = _subject_summary_table.merge(
                        _subject_roles,
                        on="subject_code",
                        how="left",
                    )
                else:
                    _subject_summary_table["role"] = "unknown"

                _comparison_errors = _comparison_df[
                    ~_comparison_df["_is_correct_for_comparison"]
                ].copy()

                if _comparison_errors.empty:
                    _top_error_pairs = _hypnotrust_comparison_pd.DataFrame(
                        {
                            "subject_code": _subject_summary_table["subject_code"],
                            "top_error_pair": "No errors",
                            "top_error_count": 0,
                        }
                    )
                else:
                    _comparison_errors["error_pair"] = (
                        _comparison_errors["true_stage"].astype(str)
                        + " → "
                        + _comparison_errors["pred_stage"].astype(str)
                    )

                    _top_error_pairs = (
                        _comparison_errors
                        .groupby(["subject_code", "error_pair"])
                        .size()
                        .reset_index(name="top_error_count")
                        .sort_values(
                            ["subject_code", "top_error_count"],
                            ascending=[True, False],
                        )
                        .drop_duplicates("subject_code")
                        .rename(columns={"error_pair": "top_error_pair"})
                    )

                _subject_summary_table = _subject_summary_table.merge(
                    _top_error_pairs,
                    on="subject_code",
                    how="left",
                )

                _subject_summary_table["top_error_pair"] = (
                    _subject_summary_table["top_error_pair"].fillna("No errors")
                )
                _subject_summary_table["top_error_count"] = (
                    _subject_summary_table["top_error_count"].fillna(0).astype(int)
                )

                _subject_summary_table = _subject_summary_table[
                    [
                        "subject_code",
                        "role",
                        "epochs",
                        "accuracy",
                        "mean_confidence",
                        "error_rate",
                        "top_error_pair",
                        "top_error_count",
                    ]
                ]

                st.dataframe(
                    _subject_summary_table,
                    use_container_width=True,
                    hide_index=True,
                )

                # Fixed: Plot accuracy and mean confidence side-by-side using Plotly for scientific correctness
                _subject_comparison_fig = go.Figure()
                _subject_comparison_fig.add_trace(
                    go.Bar(
                        x=_subject_summary_table["subject_code"],
                        y=_subject_summary_table["accuracy"],
                        name="Accuracy",
                        marker=dict(color="#3fb950", opacity=0.85),
                        hovertemplate="Subject: %{x}<br>Accuracy: %{y:.4f}<extra></extra>",
                    )
                )
                _subject_comparison_fig.add_trace(
                    go.Bar(
                        x=_subject_summary_table["subject_code"],
                        y=_subject_summary_table["mean_confidence"],
                        name="Mean Confidence",
                        marker=dict(color="#a371f7", opacity=0.85),
                        hovertemplate="Subject: %{x}<br>Mean Confidence: %{y:.4f}<extra></extra>",
                    )
                )
                _subject_comparison_fig.update_layout(
                    **PLOTLY_BASE,
                    title=dict(
                        text="Subject Comparison: Accuracy vs Mean Confidence (V5d Frozen Evaluation)",
                        font=dict(size=13, color="#8b949e"),
                    ),
                    barmode="group",
                    xaxis=dict(title="Subject Code", gridcolor="#21262d"),
                    yaxis=dict(title="Value", range=[0, 1.05], gridcolor="#21262d"),
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#21262d",
                        borderwidth=1,
                    ),
                    height=300,
                )
                st.plotly_chart(_subject_comparison_fig, use_container_width=True)

                _subject_comparison_csv = (
                    _subject_summary_table
                    .to_csv(index=False)
                    .encode("utf-8-sig")
                )

                st.download_button(
                    label="Download subject comparison CSV",
                    data=_subject_comparison_csv,
                    file_name="hypnotrust_subject_comparison.csv",
                    mime="text/csv",
                )

    # HypnoTrust demo section: safe stage distribution
    # Self-contained visualization from frozen V5d predictions.
    # No training/tuning executed.
    # ============================================================

    st.divider()

    st.subheader("Stage distribution")

    st.caption(
        "Shows how many epochs belong to each sleep stage and how many epochs the model predicted for each stage."
    )

    with st.expander("How to read stage distribution", expanded=False):
        st.markdown(
            """
            Сравняват се true и predicted counts за Wake, N1, N2, N3 и REM. 
            Ако даден predicted stage е значително по-висок от true stage, моделът може да има bias към този клас. 
            В този проект особено важен е N2-attractor ефектът.
            """
        )

    from pathlib import Path as _HypnoTrustStagePath
    import pandas as _hypnotrust_stage_pd

    # Fixed: Use project-relative path instead of hard-coded Windows path
    _hypnotrust_stage_predictions_path = Path(__file__).parent.parent / "experiments" / "v5_expanded_cross_subject_evaluation" / "v5d_frozen_external_evaluation" / "v5d_epoch_predictions.csv"
    _hypnotrust_stage_order = ["Wake", "N1", "N2", "N3", "REM"]

    if not _hypnotrust_stage_predictions_path.exists():
        st.warning("Stage distribution unavailable: predictions CSV not found.")
    else:
        _hypnotrust_stage_df = _hypnotrust_stage_pd.read_csv(
            _hypnotrust_stage_predictions_path
        )

        _stage_required_columns = {"subject_code", "true_stage", "pred_stage"}
        _stage_missing_columns = _stage_required_columns.difference(
            _hypnotrust_stage_df.columns
        )

        if _stage_missing_columns:
            st.warning(
                "Stage distribution unavailable: missing columns "
                + ", ".join(sorted(_stage_missing_columns))
            )
        else:
            _selected_subject_for_stage_distribution = None

            for _candidate_name in (
                "selected_subject",
                "subject_code",
                "current_subject",
                "subject",
            ):
                _candidate_value = locals().get(_candidate_name)
                if _candidate_value is not None:
                    _selected_subject_for_stage_distribution = str(_candidate_value)
                    break

            _available_stage_subjects = sorted(
                _hypnotrust_stage_df["subject_code"].astype(str).unique().tolist()
            )

            if _selected_subject_for_stage_distribution not in _available_stage_subjects:
                _selected_subject_for_stage_distribution = st.selectbox(
                    "Subject for stage distribution",
                    _available_stage_subjects,
                    index=0,
                    key="stage_distribution_subject_selector",
                )

            _stage_subject_df = _hypnotrust_stage_df[
                _hypnotrust_stage_df["subject_code"].astype(str)
                == _selected_subject_for_stage_distribution
            ].copy()

            if _stage_subject_df.empty:
                st.info("No stage data available for the selected subject.")
            else:
                _true_counts = (
                    _stage_subject_df["true_stage"]
                    .astype(str)
                    .value_counts()
                    .reindex(_hypnotrust_stage_order, fill_value=0)
                )

                _pred_counts = (
                    _stage_subject_df["pred_stage"]
                    .astype(str)
                    .value_counts()
                    .reindex(_hypnotrust_stage_order, fill_value=0)
                )

                _stage_distribution_table = _hypnotrust_stage_pd.DataFrame(
                    {
                        "stage": _hypnotrust_stage_order,
                        "true_epochs": _true_counts.values,
                        "predicted_epochs": _pred_counts.values,
                    }
                )

                _total_true_epochs = max(
                    int(_stage_distribution_table["true_epochs"].sum()),
                    1,
                )

                _total_pred_epochs = max(
                    int(_stage_distribution_table["predicted_epochs"].sum()),
                    1,
                )

                _stage_distribution_table["true_fraction"] = (
                    _stage_distribution_table["true_epochs"] / _total_true_epochs
                )

                _stage_distribution_table["predicted_fraction"] = (
                    _stage_distribution_table["predicted_epochs"] / _total_pred_epochs
                )

                # Fixed: Plot true and predicted epoch counts side-by-side using Plotly for scientific correctness
                _stage_distribution_fig = go.Figure()
                _stage_distribution_fig.add_trace(
                    go.Bar(
                        x=_stage_distribution_table["stage"],
                        y=_stage_distribution_table["true_epochs"],
                        name="True Epochs",
                        marker=dict(color="#79c0ff", opacity=0.85),
                        hovertemplate="Stage: %{x}<br>True Epochs: %{y}<extra></extra>",
                    )
                )
                _stage_distribution_fig.add_trace(
                    go.Bar(
                        x=_stage_distribution_table["stage"],
                        y=_stage_distribution_table["predicted_epochs"],
                        name="Predicted Epochs",
                        marker=dict(color="#7ee787", opacity=0.85),
                        hovertemplate="Stage: %{x}<br>Predicted Epochs: %{y}<extra></extra>",
                    )
                )
                _stage_distribution_fig.update_layout(
                    **PLOTLY_BASE,
                    title=dict(
                        text=f"Stage Distribution: True vs Predicted Epochs for Subject {_selected_subject_for_stage_distribution} (V5d Frozen Evaluation)",
                        font=dict(size=13, color="#8b949e"),
                    ),
                    barmode="group",
                    xaxis=dict(title="Sleep Stage", gridcolor="#21262d"),
                    yaxis=dict(title="Number of Epochs", gridcolor="#21262d"),
                    legend=dict(
                        bgcolor="#161b22",
                        bordercolor="#21262d",
                        borderwidth=1,
                    ),
                    height=300,
                )
                st.plotly_chart(_stage_distribution_fig, use_container_width=True)

                with st.expander("Open stage distribution table", expanded=False):
                    st.dataframe(
                        _stage_distribution_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    _stage_distribution_csv = (
                        _stage_distribution_table
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    )

                    st.download_button(
                        label="Download stage distribution CSV",
                        data=_stage_distribution_csv,
                        file_name=(
                            "hypnotrust_stage_distribution_"
                            + _selected_subject_for_stage_distribution
                            + ".csv"
                        ),
                        mime="text/csv",
                    )

    st.subheader("Error-pair bar chart")

    st.caption(
        "Shows the most common misclassification pairs, not just the total number of mistakes."
    )

    with st.expander("How to read error-pair chart", expanded=False):
        st.markdown(
            """
            Error-pair chart показва как точно греши моделът. 
            Пример: REM → N2 означава, че истинският stage е REM, но моделът е предсказал N2. 
            Това е по-информативно от само accuracy, защото показва failure modes.
            """
        )

    from pathlib import Path as _HypnoTrustPath
    import pandas as _hypnotrust_pd

    # Fixed: Use project-relative path instead of hard-coded Windows path
    _hypnotrust_predictions_path = Path(__file__).parent.parent / "experiments" / "v5_expanded_cross_subject_evaluation" / "v5d_frozen_external_evaluation" / "v5d_epoch_predictions.csv"

    if not _hypnotrust_predictions_path.exists():
        st.warning("Error-pair chart unavailable: predictions CSV not found.")
    else:
        _hypnotrust_error_df = _hypnotrust_pd.read_csv(_hypnotrust_predictions_path)

        _required_columns = {"subject_code", "true_stage", "pred_stage", "correct"}
        _missing_columns = _required_columns.difference(_hypnotrust_error_df.columns)

        if _missing_columns:
            st.warning(
                "Error-pair chart unavailable: missing columns "
                + ", ".join(sorted(_missing_columns))
            )
        else:
            _selected_subject_for_error_chart = None

            for _candidate_name in (
                "selected_subject",
                "subject_code",
                "current_subject",
                "subject",
            ):
                _candidate_value = locals().get(_candidate_name)
                if _candidate_value is not None:
                    _selected_subject_for_error_chart = str(_candidate_value)
                    break

            if _selected_subject_for_error_chart is not None:
                _hypnotrust_error_df = _hypnotrust_error_df[
                    _hypnotrust_error_df["subject_code"].astype(str)
                    == _selected_subject_for_error_chart
                ]

            if _hypnotrust_error_df.empty:
                st.info("No data available for the selected subject.")
            else:
                _correct_series = _hypnotrust_error_df["correct"]

                if str(_correct_series.dtype) == "bool":
                    _error_mask = ~_correct_series
                else:
                    _error_mask = ~(
                        _correct_series
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1", "yes"])
                    )

                _error_pair_source = _hypnotrust_error_df.loc[
                    _error_mask,
                    ["true_stage", "pred_stage"],
                ].copy()

                if _error_pair_source.empty:
                    st.info("No misclassified epochs for the selected subject.")
                else:
                    _error_pair_source["error_pair"] = (
                        _error_pair_source["true_stage"].astype(str)
                        + " → "
                        + _error_pair_source["pred_stage"].astype(str)
                    )

                    _error_pair_chart_data = (
                        _error_pair_source
                        .groupby("error_pair")
                        .size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False)
                        .head(10)
                    )

                    st.bar_chart(
                        _error_pair_chart_data.set_index("error_pair")["count"]
                    )

                    with st.expander("Open error-pair table", expanded=False):
                        st.dataframe(
                            _error_pair_chart_data,
                            use_container_width=True,
                            hide_index=True,
                        )



if __name__ == "__main__":
    main()
