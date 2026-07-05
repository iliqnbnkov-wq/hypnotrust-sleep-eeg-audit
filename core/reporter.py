from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from core.types import PipelineResult, ReviewQueueResult


CSV_DISCLAIMER_LINES = [
    "# HypnoTrust Research Prototype",
    "# For research and demonstration purposes only",
    "# Not for clinical use",
    "# This tool does not provide medical diagnosis",
    "# Results should be reviewed by a qualified sleep specialist",
]


def _metadata_header(extra_lines: list[str] | None = None) -> str:
    """
    Build a comment-style metadata header for exported files.

    The header keeps safety context attached to CSV exports.
    """

    lines = list(CSV_DISCLAIMER_LINES)

    if extra_lines:
        lines.extend(extra_lines)

    return "\n".join(lines) + "\n"


def build_predictions_dataframe(result: PipelineResult) -> pd.DataFrame:
    """
    Build a prediction table from PipelineResult.

    This table is safe to export in label-free User mode because it contains
    predictions and uncertainty, not accuracy or correctness claims.
    """

    n_epochs = len(result.predicted_stages)

    df = pd.DataFrame(
        {
            "epoch": list(range(n_epochs)),
            "predicted_stage": result.predicted_stages,
        }
    )

    if result.labels is not None:
        df["true_stage"] = result.labels

    uncertainty_df = result.uncertainty_df.copy().reset_index(drop=True)

    for column in uncertainty_df.columns:
        if column not in df.columns:
            df[column] = uncertainty_df[column].values

    probabilities = result.probabilities

    for class_index, stage_name in enumerate(result.stage_names):
        df[f"prob_{stage_name}"] = probabilities[:, class_index]

    epoch_length_sec = result.config_used.epoch_length_sec
    df["start_time_sec"] = df["epoch"] * epoch_length_sec
    df["end_time_sec"] = df["start_time_sec"] + epoch_length_sec

    ordered_columns = [
        "epoch",
        "start_time_sec",
        "end_time_sec",
        "predicted_stage",
    ]

    if "true_stage" in df.columns:
        ordered_columns.append("true_stage")

    remaining_columns = [col for col in df.columns if col not in ordered_columns]

    return df[ordered_columns + remaining_columns]


def predictions_to_csv_bytes(result: PipelineResult) -> bytes:
    """
    Export predictions to CSV bytes for Streamlit download_button.

    Includes disclaimer header.
    """

    df = build_predictions_dataframe(result)

    extra_lines = [
        f"# Model path: {result.config_used.model_path}",
        f"# Uncertainty policy: {result.config_used.uncertainty_policy}",
        f"# Epoch length seconds: {result.config_used.epoch_length_sec}",
        f"# Labels provided: {result.labels is not None}",
    ]

    buffer = StringIO()
    buffer.write(_metadata_header(extra_lines))
    df.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")


def review_queue_to_csv_bytes(queue: ReviewQueueResult) -> bytes:
    """
    Export review queue table to CSV bytes.

    This export contains uncertain epochs flagged for review.
    """

    extra_lines = [
        f"# Review fraction: {queue.review_fraction}",
        f"# Number reviewed: {queue.n_review}",
        f"# Number retained: {queue.n_retained}",
        f"# Total epochs: {queue.n_total}",
        f"# Confidence threshold: {queue.confidence_threshold}",
    ]

    buffer = StringIO()
    buffer.write(_metadata_header(extra_lines))
    queue.review_table.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")


def retained_epochs_to_csv_bytes(queue: ReviewQueueResult) -> bytes:
    """
    Export retained non-review epochs to CSV bytes.
    """

    extra_lines = [
        f"# Review fraction: {queue.review_fraction}",
        f"# Number reviewed: {queue.n_review}",
        f"# Number retained: {queue.n_retained}",
        f"# Total epochs: {queue.n_total}",
    ]

    buffer = StringIO()
    buffer.write(_metadata_header(extra_lines))
    queue.retained_table.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")


def metrics_to_json_like_dict(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Prepare metrics for report/export.

    Metrics may be None when no ground-truth labels exist.
    """

    if metrics is None:
        return {
            "metrics_available": False,
            "reason": "Ground-truth labels were not provided.",
        }

    return {
        "metrics_available": True,
        "metrics": metrics,
    }


def generate_html_report(
    result: PipelineResult,
    queue: ReviewQueueResult,
    metrics: dict[str, Any] | None = None,
) -> str:
    """
    Generate a simple HTML report.

    This is the MVP report format. PDF generation can come later.
    """

    predictions_df = build_predictions_dataframe(result)

    metrics_block = metrics_to_json_like_dict(metrics)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HypnoTrust Research Prototype Report</title>
</head>
<body>
    <h1>HypnoTrust Research Prototype Report</h1>

    <h2>Safety Notice</h2>
    <p>
        <strong>Research prototype only.</strong>
        This report is for research and demonstration purposes only.
        It is not for clinical use, does not provide medical diagnosis,
        and does not replace review by a qualified sleep specialist.
    </p>

    <h2>Recording Summary</h2>
    <ul>
        <li>Total epochs: {len(result.predicted_stages)}</li>
        <li>Epoch length: {result.config_used.epoch_length_sec} seconds</li>
        <li>Labels provided: {result.labels is not None}</li>
        <li>Uncertainty policy: {result.config_used.uncertainty_policy}</li>
        <li>Model path: {result.config_used.model_path}</li>
    </ul>

    <h2>Review Queue Summary</h2>
    <ul>
        <li>Review fraction: {queue.review_fraction}</li>
        <li>Epochs flagged for review: {queue.n_review}</li>
        <li>Epochs retained: {queue.n_retained}</li>
        <li>Total epochs: {queue.n_total}</li>
        <li>Confidence threshold: {queue.confidence_threshold}</li>
    </ul>

    <h2>Metrics Availability</h2>
    <pre>{metrics_block}</pre>

    <h2>First 20 Predictions</h2>
    {predictions_df.head(20).to_html(index=False)}

    <h2>First 20 Review Queue Epochs</h2>
    {queue.review_table.head(20).to_html(index=False)}
</body>
</html>
"""

    return html


def html_report_to_bytes(
    result: PipelineResult,
    queue: ReviewQueueResult,
    metrics: dict[str, Any] | None = None,
) -> bytes:
    """
    Return HTML report as bytes for Streamlit download_button.
    """

    return generate_html_report(result, queue, metrics).encode("utf-8")


def save_bytes(data: bytes, path: str | Path) -> None:
    """
    Save bytes to disk.

    Useful for non-Streamlit scripts and tests.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)