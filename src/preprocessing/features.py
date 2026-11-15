"""Causal imputations, time features, PM2.5 history and next-hour target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loading import _location_segment_ids
from .schema import MODEL_INPUT_COLUMNS, RAW_MEASUREMENT_COLUMNS, SELECTED_FEATURE_COLUMNS, CANDIDATE_FEATURE_COLUMNS, TARGET_COLUMN


def _add_causal_inputs(frame: pd.DataFrame, max_fill_hours: int) -> dict[str, int]:
    """Create short forward-filled model inputs and their audit flags in place."""
    segments = frame["location_segment_id"]
    fill_counts = {}
    for column in MODEL_INPUT_COLUMNS:
        input_column = f"{column}_input_causal"
        frame[input_column] = frame.groupby(segments, sort=False)[column].ffill(limit=max_fill_hours)
        filled_column = f"{column}_was_forward_filled"
        frame[filled_column] = frame[column].isna() & frame[input_column].notna()
        fill_counts[column] = int(frame[filled_column].sum())
    return fill_counts


def _add_time_features(frame: pd.DataFrame) -> None:
    """Create calendar components and cyclical encodings from local timestamps."""
    local_time = frame["datetime"].dt.tz_convert("Asia/Ho_Chi_Minh")
    frame["hour_of_day"] = local_time.dt.hour
    frame["day_of_week"] = local_time.dt.dayofweek
    frame["month"] = local_time.dt.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_of_day"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_of_day"] / 24)
    frame["day_of_week_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)


def _add_history_features(
    frame: pd.DataFrame,
    rolling_window_hours: int,
    rolling_min_observations: int,
) -> None:
    """Create candidate lag features and the raw-observation PM2.5 rolling mean."""
    segments = frame["location_segment_id"]
    pm25_input = "pm2_5_ug_m3_input_causal"
    for lag in (1, 3, 24):
        frame[f"pm2_5_lag_{lag}h"] = frame.groupby(segments, sort=False)[pm25_input].shift(lag)

    # Use raw measurements so causal forward-fill never inflates coverage.
    # Requiring 18/24 raw observations enforces at least 75% window coverage;
    # max_fill_hours applies only to instantaneous inputs, not this statistic.
    raw_pm25 = frame.groupby(segments, sort=False)["pm2_5_ug_m3"]
    frame["pm2_5_rolling24h_mean"] = raw_pm25.transform(
        lambda values: values.rolling(
            window=rolling_window_hours,
            min_periods=rolling_min_observations,
        ).mean()
    )
    frame["pm2_5_rolling24h_observed_count"] = raw_pm25.transform(
        lambda values: values.rolling(window=rolling_window_hours, min_periods=1).count()
    )

    lagged_measurements = [
        "pm1_ug_m3",
        "relative_humidity_pct",
        "temperature_c",
        "pm0_3_count_particles_cm3",
    ]
    for column in lagged_measurements:
        causal_column = f"{column}_input_causal"
        frame[f"{column}_lag_1h"] = frame.groupby(segments, sort=False)[causal_column].shift(1)


def _add_next_hour_target(frame: pd.DataFrame) -> None:
    """Attach the next timestamp and its observed PM2.5 target within each segment."""
    grouped_pm25 = frame.groupby("location_segment_id", sort=False)["pm2_5_ug_m3"]
    frame["target_datetime"] = frame["datetime"] + pd.Timedelta(hours=1)
    frame[TARGET_COLUMN] = grouped_pm25.shift(-1)


def prepare_hourly_data(
    frame: pd.DataFrame,
    max_causal_fill_hours: int = 2,
    rolling_window_hours: int = 24,
    rolling_min_observations: int = 18,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create causal input values, time features, lags, rolling stats, and next-hour label.

    Short gaps in instantaneous model inputs are filled forward from past observations
    only. Raw measurements remain unchanged. The rolling feature uses raw PM2.5 and
    requires a configured minimum count of observed values. PM10 is not filled, and
    the target is the observed PM2.5 value at the exact next hour. AQI is not calculated
    in this preprocessing stage.
    """
    if max_causal_fill_hours < 0:
        raise ValueError("max_causal_fill_hours must be non-negative")
    if rolling_window_hours <= 0 or not 1 <= rolling_min_observations <= rolling_window_hours:
        raise ValueError("Invalid rolling window settings")

    prepared = frame.copy().sort_values("datetime").reset_index(drop=True)
    if "location_segment_id" not in prepared:
        prepared["location_segment_id"] = _location_segment_ids(prepared)

    for column in RAW_MEASUREMENT_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    fill_counts = _add_causal_inputs(prepared, max_causal_fill_hours)
    _add_time_features(prepared)
    _add_history_features(prepared, rolling_window_hours, rolling_min_observations)
    _add_next_hour_target(prepared)

    report: dict[str, object] = {
        "max_causal_fill_hours": int(max_causal_fill_hours),
        "rolling_window_hours": int(rolling_window_hours),
        "rolling_min_observations": int(rolling_min_observations),
        "forward_filled_input_values": fill_counts,
        "features": SELECTED_FEATURE_COLUMNS,
        "candidate_features_for_diagnostics": CANDIDATE_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "pm10_excluded_from_default_features_due_to_sparse_coverage": True,
        "target_is_imputed": False,
    }
    return prepared, report
