# app/app.py
"""
HypnoTrust вЂ” Research Prototype Dashboard
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config вЂ” must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HypnoTrust",
    page_icon="рџ”¬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Core imports вЂ” hard failure: app only runs inside project environment
# ---------------------------------------------------------------------------

try:
    from core.config import get_default_pipeline_config, validate_app_paths
    from core.pipeline import run_feature_dataframe_pipeline, run_full_pipeline
    from core.review_queue import build_review_queue
    from core.reporter import predictions_to_csv_bytes, review_queue_to_csv_bytes
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
# Load pipeline config once вЂ” drives feature schema everywhere
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
        'вљ  RESEARCH PROTOTYPE</span>'
        '<span style="color:#8b949e;font-size:0.77rem;margin-left:0.6rem;">'
        'Not for clinical use. Results do not constitute medical diagnosis '
        'and must be reviewed by a qualified sleep specialist.</span></div>',
        unsafe_allow_html=True,
    )


def _mode_badge(mode: str) -> None:
    palette = {
        "Demo": ("#388bfd", "рџЋЇ  DEMO"),
        "Research": ("#a371f7", "рџ”¬  RESEARCH"),
        "User": ("#3fb950", "рџ‘¤  USER"),
        "EDF Upload": ("#f78716", "рџ§Є  EDF"),
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
        f"such as `{_TEMPORAL_FEATURE_COLUMNS[0]}` вЂ¦ "
        f"`{_TEMPORAL_FEATURE_COLUMNS[-1]}`.\n"
        f"- Baseline CSV with 13 baseline feature columns such as "
        f"`{_BASELINE_FEATURE_COLUMNS[0]}` вЂ¦ "
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
        with st.spinner("Running prediction pipelineвЂ¦"):
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
    stages = list(cm_df.columns)
    z = cm_df.values.tolist()

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=stages,
            y=stages,
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
            gridcolor="transparent",
            linecolor="#30363d",
        ),
        yaxis=dict(
            title="True",
            gridcolor="transparent",
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
        "HypnoTrust Рµ proof-of-concept dashboard Р·Р° Р°РЅР°Р»РёР· РЅР° sleep-stage predictions. "
        "РџСЂРёР»РѕР¶РµРЅРёРµС‚Рѕ РїРѕРєР°Р·РІР° РЅРµ СЃР°РјРѕ РєР°РєРІРѕ РїСЂРµРґСЃРєР°Р·РІР° РјРѕРґРµР»СЉС‚, Р° Рё РєСЉРґРµ Рµ РЅРµСЃРёРіСѓСЂРµРЅ, "
        "РєРѕРё РµРїРѕС…Рё С‚СЂСЏР±РІР° РґР° Р±СЉРґР°С‚ РїСЂРёРѕСЂРёС‚РёР·РёСЂР°РЅРё Р·Р° С‡РѕРІРµС€РєРё РїСЂРµРіР»РµРґ Рё РєР°РєРІРё СЃР° "
        "РЅР°Р№-С‡РµСЃС‚РёС‚Рµ failure modes. РќРµ Рµ РїСЂРµРґРЅР°Р·РЅР°С‡РµРЅРѕ Р·Р° СЂРµР°Р»РЅР° РјРµРґРёС†РёРЅСЃРєР° СѓРїРѕС‚СЂРµР±Р°."
    )

    with st.expander("How to interpret this dashboard", expanded=False):
        st.markdown(
            """
            **Main idea.** HypnoTrust РёР·РїРѕР»Р·РІР° РІРµС‡Рµ Р·Р°РјСЂР°Р·РµРЅРё V5d predictions. 
            РўРѕР·Рё app РЅРµ С‚СЂРµРЅРёСЂР° РјРѕРґРµР» Рё РЅРµ РїСЂРѕРјРµРЅСЏ СЂРµР·СѓР»С‚Р°С‚РёС‚Рµ.

            **Accuracy** РїРѕРєР°Р·РІР° РґРµР»Р° РїСЂР°РІРёР»РЅРѕ РєР»Р°СЃРёС„РёС†РёСЂР°РЅРё РµРїРѕС…Рё.

            **Mean confidence** РїРѕРєР°Р·РІР° СЃСЂРµРґРЅР°С‚Р° СѓРІРµСЂРµРЅРѕСЃС‚ РЅР° РјРѕРґРµР»Р°. 
            РџРѕ-РЅРёСЃРєР° СѓРІРµСЂРµРЅРѕСЃС‚ С‡РµСЃС‚Рѕ РѕР·РЅР°С‡Р°РІР° РїРѕ-РІРёСЃРѕРє СЂРёСЃРє, РЅРѕ РЅРµ РіР°СЂР°РЅС‚РёСЂР° РіСЂРµС€РєР°.

            **Hypnogram** СЃСЂР°РІРЅСЏРІР° true sleep-stage sequence СЃ predicted sequence РІСЉРІ РІСЂРµРјРµС‚Рѕ.

            **Confidence plot** РїРѕРєР°Р·РІР° РєСЉРґРµ РјРѕРґРµР»СЉС‚ Рµ РїРѕ-СЃРёРіСѓСЂРµРЅ РёР»Рё РїРѕ-РЅРµСЃРёРіСѓСЂРµРЅ РїРѕ РІСЂРµРјРµ РЅР° Р·Р°РїРёСЃР°.

            **Review queue** СЃСЉРґСЉСЂР¶Р° РЅР°Р№-РЅРёСЃРєРѕ СѓРІРµСЂРµРЅРё РµРїРѕС…Рё. 
            Р¦РµР»С‚Р° РЅРµ Рµ РјРѕРґРµР»СЉС‚ РґР° Р·Р°РјРµРЅРё СЃРїРµС†РёР°Р»РёСЃС‚, Р° РґР° РЅР°СЃРѕС‡Рё РІРЅРёРјР°РЅРёРµС‚Рѕ РєСЉРј РЅР°Р№-СЂРёСЃРєРѕРІРёС‚Рµ С‡Р°СЃС‚Рё.

            **Review queue error rate** РѕР·РЅР°С‡Р°РІР° РєР°РєСЉРІ РїСЂРѕС†РµРЅС‚ РѕС‚ РёР·Р±СЂР°РЅР°С‚Р° queue СЃР° СЂРµР°Р»РЅРё РіСЂРµС€РєРё. 
            РўРѕРІР° РЅРµ Рµ РїСЂРѕС†РµРЅС‚ РѕС‚ РІСЃРёС‡РєРё РіСЂРµС€РєРё РІ С†РµР»РёСЏ Р·Р°РїРёСЃ.

            **Retained accuracy** Рµ accuracy РІСЉСЂС…Сѓ РµРїРѕС…РёС‚Рµ, РєРѕРёС‚Рѕ РѕСЃС‚Р°РІР°С‚ РёР·РІСЉРЅ review queue.

            **Error capture fraction** РїРѕРєР°Р·РІР° РєР°РєРІР° С‡Р°СЃС‚ РѕС‚ РІСЃРёС‡РєРё РіСЂРµС€РєРё СЃР° РїРѕРїР°РґРЅР°Р»Рё РІ review queue.

            **Stage distribution** СЃСЂР°РІРЅСЏРІР° true Рё predicted Р±СЂРѕР№ РµРїРѕС…Рё Р·Р° Wake, N1, N2, N3 Рё REM. 
            РђРєРѕ predicted N2 Рµ РїСЂРµРєР°Р»РµРЅРѕ РІРёСЃРѕРє, С‚РѕРІР° РјРѕР¶Рµ РґР° РїРѕРєР°Р·РІР° N2-attractor behavior.

            **Error-pair chart** РїРѕРєР°Р·РІР° РєР°Рє С‚РѕС‡РЅРѕ РіСЂРµС€Рё РјРѕРґРµР»СЉС‚, РЅР°РїСЂРёРјРµСЂ REM в†’ N2 РёР»Рё N3 в†’ N2.

            **Subject comparison** РїРѕРєР°Р·РІР° РґР°Р»Рё РїРѕРІРµРґРµРЅРёРµС‚Рѕ РЅР° РјРѕРґРµР»Р° Рµ СЃС‚Р°Р±РёР»РЅРѕ РјРµР¶РґСѓ СЂР°Р·Р»РёС‡РЅРё subjects.
            """
        )

    _section_header("Hypnogram", badge="PREDICTED")

    st.caption(
        "Hypnogram compares the reference sleep-stage sequence with the model prediction across time."
    )

    with st.expander("How to read the hypnogram", expanded=False):
        st.markdown(
            """
            Hypnogram РїРѕРєР°Р·РІР° sleep stages РІСЉРІ РІСЂРµРјРµС‚Рѕ. 
            РљРѕРіР°С‚Рѕ true Рё predicted Р»РёРЅРёРёС‚Рµ СЃРµ СЂР°Р·РјРёРЅР°РІР°С‚, РјРѕРґРµР»СЉС‚ РЅРµ СЃР»РµРґРІР° РїСЂР°РІРёР»РЅРѕ СЂРµС„РµСЂРµРЅС‚РЅР°С‚Р° РїРѕСЃР»РµРґРѕРІР°С‚РµР»РЅРѕСЃС‚. 
            РџСЂРѕРґСЉР»Р¶РёС‚РµР»РЅРё СЂР°Р·РјРёРЅР°РІР°РЅРёСЏ СЃР° РїРѕ-РІР°Р¶РЅРё РѕС‚ РµРґРёРЅРёС‡РЅРё РіСЂРµС€РєРё.
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
                "рџ“Љ No verified **Stage** column found. "
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

    with st.expander("рџ“‹ Full prediction table", expanded=False):
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
        "в¬‡ Predictions CSV",
        data=pred_bytes,
        file_name="hypnotrust_predictions.csv",
        mime="text/csv",
    )

    ex2.download_button(
        "в¬‡ Review Queue CSV",
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
            '<span style="font-size:1.6rem;">рџ”¬</span>'
            '<div style="font-size:1.1rem;font-weight:700;color:#e6edf3;'
            'margin-top:0.2rem;">HypnoTrust</div>'
            '<div style="font-size:0.7rem;color:#6e7681;letter-spacing:0.06em;">'
            'RESEARCH PROTOTYPE v0.1</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        mode = st.radio(
            "Mode",
            options=["User", "Demo", "EDF Upload", "Research"],
            index=1,
            help=(
                "User: upload feature CSV, predictions only.\n"
                "Demo: synthetic profiles through the real model.\n"
                "Research: full metrics when verified labels are present."
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
            f'<b style="color:#8b949e;">Stages:</b> Wake В· N1 В· N2 В· N3 В· REM<br>'
            f'<b style="color:#8b949e;">Epoch:</b> 30 s<br>'
            f'<b style="color:#8b949e;">Features:</b> '
            f'{len(_TEMPORAL_FEATURE_COLUMNS)} temporal</div>',
            unsafe_allow_html=True,
        )

    return mode, params




# ---------------------------------------------------------------------------
# EDF Upload mode
# ---------------------------------------------------------------------------

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
        'MNE EDF reader в†’ feature extraction в†’ Temporal v2 model в†’ uncertainty в†’ review queue. '
        '<b style="color:#f78716;">Experimental research mode only.</b></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Use this only with compatible PSG EDF files containing the expected Sleep-EDF style channels "
        "such as Fpz-Cz, Pz-Oz and horizontal EOG. Hypnogram/annotation EDF is optional.",
        icon="в„№пёЏ",
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

    if st.button("в–¶  Run EDF Pipeline", type="primary"):
        psg_path = None
        hypnogram_path = None

        try:
            import mne

            psg_path = _save_uploaded_edf(psg_file, ".edf")

            with st.spinner("Reading PSG EDF with MNEвЂ¦"):
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

            with st.spinner("Running full EDF pipelineвЂ¦"):
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

            st.session_state["edf_result"] = result
            st.session_state["edf_queue"] = queue
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

    if st.button("в–¶  Generate Demo", type="primary"):
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
            f'Profile: <b style="color:#79c0ff;">{meta["profile"]}</b> В· '
            f'{meta["n_epochs"]} epochs В· seed {meta["seed"]}</div>',
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
        f'&nbsp;вЂў <b style="color:#e6edf3;">Temporal v2 CSV</b>: '
        f'{n_features} feature columns '
        f'(e.g. <code>{_TEMPORAL_FEATURE_COLUMNS[0]}</code> вЂ¦ '
        f'<code>{_TEMPORAL_FEATURE_COLUMNS[-1]}</code>)<br>'
        f'&nbsp;вЂў <b style="color:#e6edf3;">Baseline CSV</b>: '
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
        f"вњ“ CSV schema detected: `{schema_name}`. "
        f"Rows: {len(feature_df)}. "
        f"Features: {len(_TEMPORAL_FEATURE_COLUMNS)}."
    )

    if st.button("в–¶  Run Prediction", type="primary"):
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
        f"вњ“ CSV schema detected: `{schema_name}`. "
        f"Rows: {len(feature_df)}. "
        f"Features: {len(_TEMPORAL_FEATURE_COLUMNS)}."
    )

    if labels is not None:
        st.success(
            f"вњ“ Verified Stage labels detected "
            f"({len(set(labels))} classes, {len(labels)} epochs). "
            f"Metrics unlocked."
        )
    else:
        st.warning(
            "No verified **Stage** column found. "
            "Showing predictions and uncertainty only."
        )

    if st.button("в–¶  Run Analysis", type="primary"):
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
        '<span style="font-size:2rem;">рџ”¬</span>'
        '<div>'
        '<div style="font-size:1.45rem;font-weight:700;color:#e6edf3;'
        'line-height:1.1;">HypnoTrust</div>'
        '<div style="font-size:0.76rem;color:#6e7681;letter-spacing:0.04em;">'
        'Uncertainty-Aware Sleep Stage Classification В· Research Prototype'
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
            РўР°Р·Рё С‚Р°Р±Р»РёС†Р° РїРѕРєР°Р·РІР° РґР°Р»Рё РјРѕРґРµР»СЉС‚ СЃРµ РґСЉСЂР¶Рё СЃС‚Р°Р±РёР»РЅРѕ РїСЂРё СЂР°Р·Р»РёС‡РЅРё subjects. 
            Р“РѕР»РµРјРё СЂР°Р·Р»РёРєРё РІ accuracy РёР»Рё confidence СЃР° РІР°Р¶РЅРё, Р·Р°С‰РѕС‚Рѕ sleep EEG РёРјР° СЃРёР»РЅР° РёРЅРґРёРІРёРґСѓР°Р»РЅР° РІР°СЂРёР°Р±РёР»РЅРѕСЃС‚. 
            `Top error pair` РїРѕРєР°Р·РІР° РЅР°Р№-С‡РµСЃС‚РёСЏ С‚РёРї РіСЂРµС€РєР° Р·Р° РґР°РґРµРЅ subject.
            """
        )

    from pathlib import Path as _HypnoTrustComparisonPath
    import pandas as _hypnotrust_comparison_pd

    _hypnotrust_comparison_predictions_path = _HypnoTrustComparisonPath(
        r".\experiments\v5_expanded_cross_subject_evaluation\v5d_frozen_external_evaluation\v5d_epoch_predictions.csv"
    )

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
                        + " в†’ "
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

                st.bar_chart(
                    _subject_summary_table
                    .set_index("subject_code")[["accuracy", "mean_confidence"]]
                )

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
            РЎСЂР°РІРЅСЏРІР°С‚ СЃРµ true Рё predicted counts Р·Р° Wake, N1, N2, N3 Рё REM. 
            РђРєРѕ РґР°РґРµРЅ predicted stage Рµ Р·РЅР°С‡РёС‚РµР»РЅРѕ РїРѕ-РІРёСЃРѕРє РѕС‚ true stage, РјРѕРґРµР»СЉС‚ РјРѕР¶Рµ РґР° РёРјР° bias РєСЉРј С‚РѕР·Рё РєР»Р°СЃ. 
            Р’ С‚РѕР·Рё РїСЂРѕРµРєС‚ РѕСЃРѕР±РµРЅРѕ РІР°Р¶РµРЅ Рµ N2-attractor РµС„РµРєС‚СЉС‚.
            """
        )

    from pathlib import Path as _HypnoTrustStagePath
    import pandas as _hypnotrust_stage_pd

    _hypnotrust_stage_predictions_path = _HypnoTrustStagePath(r".\experiments\v5_expanded_cross_subject_evaluation\v5d_frozen_external_evaluation\v5d_epoch_predictions.csv")
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

                st.bar_chart(
                    _stage_distribution_table
                    .set_index("stage")[["true_epochs", "predicted_epochs"]]
                )

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
            Error-pair chart РїРѕРєР°Р·РІР° РєР°Рє С‚РѕС‡РЅРѕ РіСЂРµС€Рё РјРѕРґРµР»СЉС‚. 
            РџСЂРёРјРµСЂ: REM в†’ N2 РѕР·РЅР°С‡Р°РІР°, С‡Рµ РёСЃС‚РёРЅСЃРєРёСЏС‚ stage Рµ REM, РЅРѕ РјРѕРґРµР»СЉС‚ Рµ РїСЂРµРґСЃРєР°Р·Р°Р» N2. 
            РўРѕРІР° Рµ РїРѕ-РёРЅС„РѕСЂРјР°С‚РёРІРЅРѕ РѕС‚ СЃР°РјРѕ accuracy, Р·Р°С‰РѕС‚Рѕ РїРѕРєР°Р·РІР° failure modes.
            """
        )

    from pathlib import Path as _HypnoTrustPath
    import pandas as _hypnotrust_pd

    _hypnotrust_predictions_path = _HypnoTrustPath(r".\experiments\v5_expanded_cross_subject_evaluation\v5d_frozen_external_evaluation\v5d_epoch_predictions.csv")

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
                        + " в†’ "
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

