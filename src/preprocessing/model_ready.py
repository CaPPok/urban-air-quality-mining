"""Chronological split and feature diagnostics for the prepared dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    CANDIDATE_FEATURE_COLUMNS, EXCLUDED_FEATURE_REASONS, FEATURE_SELECTION_REASONS,
    MODELING_LOCATION, SELECTED_FEATURE_COLUMNS, TARGET_COLUMN, _feature_reason,
)


def _assign_time_splits(
    site: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.Timestamp]:
    """Label each row by a chronological split using target timestamps."""
    target_times = site["target_datetime"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if len(target_times) < 3:
        raise ValueError("Not enough target timestamps to create temporal train/validation/test splits")

    train_end = min(max(int(len(target_times) * train_fraction), 1), len(target_times) - 2)
    validation_end = min(
        max(int(len(target_times) * (train_fraction + validation_fraction)), train_end + 1),
        len(target_times) - 1,
    )
    validation_start = target_times.iloc[train_end]
    test_start = target_times.iloc[validation_end]
    site = site.copy()
    site["split"] = np.select(
        [site["target_datetime"] < validation_start, site["target_datetime"] < test_start],
        ["train", "validation"],
        default="test",
    )
    return site, target_times, validation_start, test_start


def _feature_correlation_diagnostics(
    training_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Calculate candidate-to-target and candidate-to-candidate correlations on train."""
    target_pearson = training_rows.corr(method="pearson", min_periods=30)[TARGET_COLUMN]
    target_spearman = training_rows.corr(method="spearman", min_periods=30)[TARGET_COLUMN]
    feature_matrix = training_rows[CANDIDATE_FEATURE_COLUMNS].corr(method="pearson", min_periods=30)

    records = []
    for feature in CANDIDATE_FEATURE_COLUMNS:
        paired = training_rows[[feature, TARGET_COLUMN]].dropna()
        records.append(
            {
                "feature": feature,
                "train_pairwise_rows": int(len(paired)),
                "pearson_with_target": float(target_pearson[feature]) if pd.notna(target_pearson[feature]) else np.nan,
                "spearman_with_target": float(target_spearman[feature]) if pd.notna(target_spearman[feature]) else np.nan,
                "selected_for_default_model": feature in SELECTED_FEATURE_COLUMNS,
                "selection_note": _feature_reason(feature),
            }
        )
    return pd.DataFrame(records), feature_matrix, target_pearson, target_spearman


def _complete_model_rows(site: pd.DataFrame) -> pd.DataFrame:
    """Keep only baseline inputs, observed target, and split metadata."""
    complete = site[SELECTED_FEATURE_COLUMNS + [TARGET_COLUMN]].notna().all(axis=1)
    columns = [
        *SELECTED_FEATURE_COLUMNS,
        TARGET_COLUMN,
        "split",
    ]
    return site.loc[complete, columns].reset_index(drop=True)


def prepare_model_ready_dataset(
    prepared: pd.DataFrame,
    location: str = MODELING_LOCATION,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select a compact feature set and create complete, chronological model rows.

    Feature correlations are measured only in the training time window. Validation
    and test periods are not used for feature selection or imputation. The returned
    model table has no missing feature or target values and includes a split label.
    """
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one")
    missing = sorted(set(CANDIDATE_FEATURE_COLUMNS + [TARGET_COLUMN, "datetime", "target_datetime", "location"]) - set(prepared.columns))
    if missing:
        raise ValueError(f"Prepared data is missing modeling columns: {missing}")

    site = prepared.loc[prepared["location"].eq(location)].copy().sort_values("datetime")
    if site.empty:
        available = sorted(prepared["location"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Modeling location {location!r} not found. Available locations: {available}")
    location_ids = site["location_id"].dropna().unique()
    location_id = location_ids[0] if len(location_ids) == 1 else None

    site, target_times, validation_start, test_start = _assign_time_splits(
        site,
        train_fraction,
        validation_fraction,
    )
    training_observed = site.loc[
        (site["split"] == "train") & site[TARGET_COLUMN].notna(),
        CANDIDATE_FEATURE_COLUMNS + [TARGET_COLUMN],
    ]
    if training_observed[TARGET_COLUMN].notna().sum() < 30:
        raise ValueError("Too few observed training targets to compute feature correlations")

    correlation_table, feature_correlation_matrix, target_corr, _ = _feature_correlation_diagnostics(
        training_observed
    )
    model_rows = _complete_model_rows(site)

    split_rows = model_rows.groupby("split", sort=False).size().to_dict()
    report: dict[str, object] = {
        "modeling_location": location,
        "modeling_location_id": int(location_id) if location_id is not None else None,
        "included_location_names": sorted(site["location"].dropna().astype(str).unique().tolist()),
        "included_location_segments": int(site["location_segment_id"].nunique()),
        "excluded_location_segments": int(prepared["location_segment_id"].nunique() - site["location_segment_id"].nunique()),
        "location_segments_are_separately_processed": True,
        "feature_count": len(SELECTED_FEATURE_COLUMNS),
        "features": SELECTED_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "audit_columns_kept_in_preprocessed_file": [
            "location_segment_id",
            "datetime",
            "target_datetime",
            "pm2_5_ug_m3_was_forward_filled",
            "temperature_c_was_forward_filled",
            "pm2_5_rolling24h_observed_count",
        ],
        "candidate_feature_count": len(CANDIDATE_FEATURE_COLUMNS),
        "selection_method": "Provisional domain-informed baseline; Pearson/Spearman correlations are descriptive only and computed using training rows.",
        "train_only_correlation_checks": {
            "pm2_5_current_vs_target_pearson": float(target_corr["pm2_5_ug_m3_input_causal"]),
            "pm1_current_vs_target_pearson": float(target_corr["pm1_ug_m3_input_causal"]),
            "pm0_3_current_vs_target_pearson": float(target_corr["pm0_3_count_particles_cm3_input_causal"]),
            "pm2_5_current_vs_pm1_current_pearson": float(feature_correlation_matrix.loc["pm2_5_ug_m3_input_causal", "pm1_ug_m3_input_causal"]),
            "pm2_5_current_vs_pm0_3_current_pearson": float(feature_correlation_matrix.loc["pm2_5_ug_m3_input_causal", "pm0_3_count_particles_cm3_input_causal"]),
            "temperature_vs_humidity_pearson": float(feature_correlation_matrix.loc["temperature_c_input_causal", "relative_humidity_pct_input_causal"]),
            "pm2_5_current_vs_lag1_pearson": float(feature_correlation_matrix.loc["pm2_5_ug_m3_input_causal", "pm2_5_lag_1h"]),
        },
        "selection_reasons": FEATURE_SELECTION_REASONS,
        "excluded_feature_reasons": EXCLUDED_FEATURE_REASONS,
        "split_method": "Chronological by target_datetime; no shuffling. Split boundaries are based on the hourly timeline before removing incomplete rows.",
        "scaling_policy": "No scaler is fit globally. Tree models can use these numeric features directly; fit a scaler on train rows only inside any scale-sensitive model pipeline.",
        "train_fraction_by_time": float(train_fraction),
        "validation_fraction_by_time": float(validation_fraction),
        "test_fraction_by_time": round(float(1 - train_fraction - validation_fraction), 10),
        "training_window_start_target_datetime": target_times.iloc[0].isoformat(),
        "training_window_end_exclusive_target_datetime": validation_start.isoformat(),
        "validation_start_target_datetime": validation_start.isoformat(),
        "test_start_target_datetime": test_start.isoformat(),
        "rows_in_selected_location_before_filtering": int(len(site)),
        "rows_excluded_from_modeling_as_other_locations": int(len(prepared) - len(site)),
        "rows_with_observed_target": int(site[TARGET_COLUMN].notna().sum()),
        "model_ready_rows_after_complete_case_filter": int(len(model_rows)),
        "rows_dropped_for_missing_selected_features_or_target": int(len(site) - len(model_rows)),
        "model_ready_rows_by_split": {str(key): int(value) for key, value in split_rows.items()},
        "no_missing_values_in_model_features_or_target": bool(not model_rows[SELECTED_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()),
    }
    return model_rows, correlation_table, feature_correlation_matrix, report
