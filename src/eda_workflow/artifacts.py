"""Write EDA tables and assemble the run summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocessing.schema import SELECTED_FEATURE_COLUMNS


def save_profile_tables(profile: dict[str, object], metrics_dir: Path) -> None:
    """Write the summaries that describe the raw hourly data."""
    tables = {
        "numeric_summary": "numeric_summary.csv",
        "location_segments": "location_segments.csv",
        "monthly_missing_pct": "monthly_missing_pct.csv",
        "outlier_summary_by_segment": "outlier_summary_by_segment.csv",
        "outlier_events_by_segment": "outlier_events_by_segment.csv",
        "missing_gap_summary_by_segment": "missing_gap_summary_by_segment.csv",
    }
    for name, filename in tables.items():
        profile[name].to_csv(metrics_dir / filename, index=name != "monthly_missing_pct")


def save_preprocessing_outputs(
    prepared: pd.DataFrame,
    model_ready: pd.DataFrame,
    feature_correlations: pd.DataFrame,
    feature_correlation_matrix: pd.DataFrame,
    output_path: Path,
    model_output_path: Path,
    metrics_dir: Path,
) -> None:
    """Write prepared data, model-ready rows and train-only feature diagnostics."""
    prepared.to_csv(output_path, index=False)
    model_ready.to_csv(model_output_path, index=False)
    feature_correlations.to_csv(metrics_dir / "feature_target_correlations_train.csv", index=False)
    feature_correlation_matrix.to_csv(metrics_dir / "feature_correlation_matrix_train.csv")


def build_pipeline_summary(
    profile: dict[str, object],
    load_report: dict[str, object],
    preparation_report: dict[str, object],
    modeling_report: dict[str, object],
    figures: dict[str, Path],
    output_path: Path,
    model_output_path: Path,
    metrics_dir: Path,
) -> dict[str, object]:
    """Collect run policies, result locations and short artifact descriptions."""
    return {
        "overview": profile["overview"],
        "gap_summary": profile["gap_summary"],
        "load_validation": load_report,
        "preprocessing": preparation_report,
        "modeling_preparation": modeling_report,
        "aqi_policy": "This preprocessing stage prepares the observed next-hour PM2.5 regression target only. Calculate the selected AQI/sub-index after PM2.5 prediction, using the chosen standard and its required historical window.",
        "outlier_policy": "IQR fences are calculated per location segment and stored as diagnostic summaries/events only; no observed values are deleted, clipped, or smoothed automatically.",
        "outlier_event_count": int(len(profile["outlier_events_by_segment"])),
        "missing_value_policy": "Causal forward fill is limited to two hours for instantaneous model inputs. Longer gaps and missing next-hour targets remain missing. The 24-hour PM2.5 rolling feature uses raw observations only and requires at least 18 observed hours.",
        "quality_audit_files": {
            "missing_gap_summary": str(metrics_dir / "missing_gap_summary_by_segment.csv"),
            "outlier_summary": str(metrics_dir / "outlier_summary_by_segment.csv"),
            "outlier_events": str(metrics_dir / "outlier_events_by_segment.csv"),
        },
        "artifact_guide": {
            "numeric_summary.csv": "Coverage and descriptive statistics by raw measurement.",
            "location_segments.csv": "Location metadata periods retained separately in the source.",
            "monthly_missing_pct.csv": "Monthly missing percentage by raw measurement.",
            "missing_gap_summary_by_segment.csv": "Missing-run counts and longest run by segment and measurement.",
            "outlier_summary_by_segment.csv": "IQR fences and flagged counts by segment; flags are for review only.",
            "outlier_events_by_segment.csv": "Rows flagged by IQR for manual review; values are not changed.",
            "feature_target_correlations_train.csv": "Train-only candidate correlations with the target and selection notes.",
            "feature_correlation_matrix_train.csv": "Pairwise candidate correlations on train rows.",
            "eda_preprocessing_summary.json": "Run summary, policies, split counts and artifact descriptions.",
        },
        "figure_guide": {
            "monthly_missingness.png": "Monthly changes in missing measurements.",
            "pm25_timeline.png": "Observed hourly PM2.5 over time and metadata segments.",
            "pm25_by_location.png": "PM2.5 distribution by metadata segment.",
            "pm25_diurnal_profile.png": "Average PM2.5 by local hour of day.",
            "correlation_heatmap.png": "Pairwise correlation between raw measurements, not feature importance.",
            "feature_target_correlations_train.png": "Train-only candidate correlations with the target; descriptive only.",
            "feature_correlation_matrix_train.png": "Train-only correlations among candidate inputs; baseline features are marked with an asterisk.",
        },
        "weather_fields_unavailable_in_source": ["pressure", "dew_point", "wind_speed", "wind_direction", "rain"],
        "processed_output": str(output_path),
        "model_ready_output": str(model_output_path),
        "selected_model_features": SELECTED_FEATURE_COLUMNS,
        "eda_figures": {name: str(path) for name, path in figures.items()},
    }
