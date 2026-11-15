"""Run the OpenAQ CMT8 EDA and preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_workflow.plots import (
    plot_eda,
    plot_feature_correlation_matrix,
    plot_feature_target_correlations,
)
from eda_workflow.profile import build_profile
from eda_workflow.artifacts import (
    build_pipeline_summary,
    save_preprocessing_outputs,
    save_profile_tables,
)
from preprocessing.preprocess import (
    MODELING_LOCATION, SELECTED_FEATURE_COLUMNS, load_hourly_data,
    prepare_hourly_data, prepare_model_ready_dataset,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    artifacts_dir: str | Path,
    max_causal_fill_hours: int = 2,
    model_output_path: str | Path | None = None,
    modeling_location: str = MODELING_LOCATION,
) -> dict[str, object]:
    """Run EDA, save report figures/tables, and write the processed hourly dataset."""
    input_path, output_path, artifacts_dir = Path(input_path), Path(output_path), Path(artifacts_dir)
    if model_output_path is None:
        model_output_path = output_path.parent / "openaq_cmt8_model_ready.csv"
    model_output_path = Path(model_output_path)
    metrics_dir = artifacts_dir / "metrics"
    figures_dir = artifacts_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    raw, load_report = load_hourly_data(input_path)
    profile = build_profile(raw)
    figures = plot_eda(raw, figures_dir)
    save_profile_tables(profile, metrics_dir)

    prepared, preparation_report = prepare_hourly_data(raw, max_causal_fill_hours=max_causal_fill_hours)
    model_ready, feature_correlations, correlation_matrix, modeling_report = prepare_model_ready_dataset(
        prepared,
        location=modeling_location,
    )
    figures["feature_target_correlations_train"] = plot_feature_target_correlations(
        feature_correlations,
        figures_dir,
    )
    figures["feature_correlation_matrix_train"] = plot_feature_correlation_matrix(
        correlation_matrix,
        figures_dir,
    )
    save_preprocessing_outputs(
        prepared,
        model_ready,
        feature_correlations,
        correlation_matrix,
        output_path,
        model_output_path,
        metrics_dir,
    )
    result = build_pipeline_summary(
        profile,
        load_report,
        preparation_report,
        modeling_report,
        figures,
        output_path,
        model_output_path,
        metrics_dir,
    )
    (metrics_dir / "eda_preprocessing_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "openaq_cmt8_2y_hourly.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "openaq_cmt8_preprocessed.csv",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--max-causal-fill-hours", type=int, default=2)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "openaq_cmt8_model_ready.csv",
    )
    parser.add_argument("--modeling-location", default=MODELING_LOCATION)
    args = parser.parse_args()

    result = run_pipeline(
        args.input,
        args.output,
        args.artifacts_dir,
        args.max_causal_fill_hours,
        args.model_output,
        args.modeling_location,
    )
    print(f"Input rows: {result['overview']['rows']:,}")
    print(f"Hours without measurements: {result['overview']['hours_with_no_measurement']:,}")
    print(f"Processed file: {result['processed_output']}")
    print(f"Model-ready file: {result['model_ready_output']}")
    print(f"Selected features: {len(result['selected_model_features'])}")
    print(f"Model-ready rows: {result['modeling_preparation']['model_ready_rows_after_complete_case_filter']:,}")
    print(f"EDA report: {args.artifacts_dir / 'metrics' / 'eda_preprocessing_summary.json'}")


if __name__ == "__main__":
    main()
