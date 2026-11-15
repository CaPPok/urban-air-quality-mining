"""Dataset columns and feature-selection notes for the CMT8 pipeline."""

from __future__ import annotations

RAW_MEASUREMENT_COLUMNS = [
    "pm1_ug_m3",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
    "relative_humidity_pct",
    "temperature_c",
    "pm0_3_count_particles_cm3",
]

# PM10 is retained for analysis, but its coverage is too low for a default model input.
MODEL_INPUT_COLUMNS = [
    "pm1_ug_m3",
    "pm2_5_ug_m3",
    "relative_humidity_pct",
    "temperature_c",
    "pm0_3_count_particles_cm3",
]

# Compact default feature set for one-hour-ahead PM2.5 regression. The candidate
# set remains available for train-only correlation diagnostics, not as model input.
SELECTED_FEATURE_COLUMNS = [
    "pm2_5_ug_m3_input_causal",
    "pm2_5_rolling24h_mean",
    "temperature_c_input_causal",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
CANDIDATE_FEATURE_COLUMNS = [
    "pm2_5_ug_m3_input_causal",
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_24h",
    "pm2_5_rolling24h_mean",
    "pm1_ug_m3_input_causal",
    "pm1_ug_m3_lag_1h",
    "relative_humidity_pct_input_causal",
    "relative_humidity_pct_lag_1h",
    "temperature_c_input_causal",
    "temperature_c_lag_1h",
    "pm0_3_count_particles_cm3_input_causal",
    "pm0_3_count_particles_cm3_lag_1h",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]
TARGET_COLUMN = "target_pm2_5_next_hour"
MODELING_LOCATION = "CMT8-3246311"

REQUIRED_COLUMNS = ["datetime", "location_id", "location", "latitude", "longitude", *RAW_MEASUREMENT_COLUMNS]
METADATA_COLUMNS = ["location_id", "location", "latitude", "longitude"]

FEATURE_SELECTION_REASONS = {
    "pm2_5_ug_m3_input_causal": "Latest observed or short-gap-filled PM2.5 value; directly available at prediction time.",
    "pm2_5_rolling24h_mean": "Recent PM2.5 background; uses raw observations and requires at least 18 of 24 hours.",
    "temperature_c_input_causal": "Available meteorological signal; kept as a candidate despite weak marginal correlation.",
    "hour_sin/hour_cos": "Pair encoding within-day periodicity without treating clock hours as a linear scale.",
    "month_sin/month_cos": "Pair encoding annual seasonality; stability should be checked on later time periods.",
}

EXCLUDED_FEATURE_REASONS = {
    "pm1_ug_m3_input_causal and pm0_3_count_particles_cm3_input_causal": "Strongly overlap with concurrent PM2.5; omitted from the compact baseline, not proven unhelpful.",
    "pm2_5_lag_1h": "Redundant with current PM2.5 in the training period and has a weaker marginal target association.",
    "pm2_5_lag_3h and pm2_5_lag_24h": "Omitted from the compact baseline; rolling mean represents recent history.",
    "pm1_ug_m3_lag_1h and pm0_3_count_particles_cm3_lag_1h": "Lagged particulate measures overlap with PM2.5 history and are omitted from the compact baseline.",
    "relative_humidity_pct_input_causal": "Strongly correlated with temperature and has weak marginal target correlation in this dataset.",
    "relative_humidity_pct_lag_1h": "Lagged humidity has a weak marginal target association and is redundant with current humidity.",
    "temperature_c_lag_1h": "Omitted to keep one compact meteorological input; its incremental value is not established.",
    "day_of_week_sin/day_of_week_cos": "Weak marginal association in the training period; left out of the compact baseline.",
    "pm10_ug_m3": "Very sparse coverage; retained for EDA only.",
}


def _feature_reason(feature: str) -> str:
    """Return the brief rationale for a selected or omitted diagnostic feature."""
    reasons = (
        *FEATURE_SELECTION_REASONS.items(),
        *EXCLUDED_FEATURE_REASONS.items(),
    )
    for feature_group, reason in reasons:
        names = feature_group.replace(" and ", "/").split("/")
        if feature in names:
            return reason
    return "Candidate retained for descriptive correlation analysis only."


