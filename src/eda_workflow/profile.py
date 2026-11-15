"""Tabular summaries for coverage, metadata, missing runs and IQR flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.schema import MODEL_INPUT_COLUMNS, RAW_MEASUREMENT_COLUMNS, SELECTED_FEATURE_COLUMNS


def _empty_measurement_gaps(frame: pd.DataFrame) -> list[int]:
    """Return lengths of consecutive hours with no measured pollutant or weather value."""
    empty = frame[RAW_MEASUREMENT_COLUMNS].isna().all(axis=1)
    groups = empty.groupby(empty.ne(empty.shift()).cumsum())
    return [int(group.sum()) for _, group in groups if bool(group.iloc[0])]


def _numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage and distribution for each raw measurement."""
    records = []
    for column in RAW_MEASUREMENT_COLUMNS:
        values = frame[column].dropna()
        missing = int(frame[column].isna().sum())
        record = {
            "column": column,
            "observed": int(values.size),
            "missing": missing,
            "missing_pct": float(100 * missing / len(frame)) if len(frame) else 0.0,
        }
        if not values.empty:
            record.update(
                {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _location_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe each contiguous location-metadata period."""
    records = []
    for segment_id, group in frame.groupby("location_segment_id", sort=True):
        records.append(
            {
                "location_segment_id": int(segment_id),
                "location": str(group["location"].iloc[0]),
                "latitude": float(group["latitude"].dropna().iloc[0]),
                "longitude": float(group["longitude"].dropna().iloc[0]),
                "start_datetime": group["datetime"].min().isoformat(),
                "end_datetime": group["datetime"].max().isoformat(),
                "hour_rows": int(len(group)),
                "pm2_5_observed": int(group["pm2_5_ug_m3"].notna().sum()),
                "pm10_observed": int(group["pm10_ug_m3"].notna().sum()),
            }
        )
    return pd.DataFrame(records)


def _monthly_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly missing percentages for each raw measurement."""
    by_month = frame.assign(month=frame["datetime"].dt.strftime("%Y-%m"))
    return by_month.groupby("month")[RAW_MEASUREMENT_COLUMNS].agg(lambda values: 100 * values.isna().mean())


def _missing_run_lengths(values: pd.Series) -> list[int]:
    missing = values.isna()
    groups = missing.groupby(missing.ne(missing.shift()).cumsum())
    return [int(group.sum()) for _, group in groups if bool(group.iloc[0])]


def _segment_outlier_records(
    frame: pd.DataFrame,
    segment_id: int,
    segment: pd.DataFrame,
    column: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Find IQR flags for one measurement in one metadata segment."""
    values = segment[column]
    observed = values.dropna()
    if observed.empty:
        q1 = q3 = iqr = lower = upper = np.nan
        outlier_mask = pd.Series(False, index=values.index)
    else:
        q1, q3 = float(observed.quantile(0.25)), float(observed.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = values.notna() & ((values < lower) | (values > upper))

    location = str(segment["location"].iloc[0])
    summary = {
        "location_segment_id": int(segment_id),
        "location": location,
        "measurement": column,
        "observed": int(observed.size),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "iqr_outlier_count": int(outlier_mask.sum()),
    }
    events = []
    for row_index in values.index[outlier_mask]:
        value = float(frame.at[row_index, column])
        events.append(
            {
                "datetime": frame.at[row_index, "datetime"].isoformat(),
                "location_segment_id": int(segment_id),
                "location": location,
                "measurement": column,
                "value": value,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "direction": "below" if value < lower else "above",
            }
        )
    return summary, events


def _segment_missing_gap_record(segment_id: int, segment: pd.DataFrame, column: str) -> dict[str, object]:
    """Summarize missing hours and consecutive missing runs for one segment/measurement."""
    values = segment[column]
    gaps = _missing_run_lengths(values)
    return {
        "location_segment_id": int(segment_id),
        "location": str(segment["location"].iloc[0]),
        "measurement": column,
        "hour_rows": int(len(segment)),
        "missing_hours": int(values.isna().sum()),
        "missing_pct": float(100 * values.isna().mean()),
        "missing_gap_count": len(gaps),
        "longest_missing_gap_hours": max(gaps, default=0),
    }


def _segment_quality_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build IQR summary, IQR event list and missing-run summary by segment."""
    outlier_summaries = []
    outlier_events = []
    missing_gaps = []
    for segment_id, segment in frame.groupby("location_segment_id", sort=True):
        for column in RAW_MEASUREMENT_COLUMNS:
            summary, events = _segment_outlier_records(frame, segment_id, segment, column)
            outlier_summaries.append(summary)
            outlier_events.extend(events)
            missing_gaps.append(_segment_missing_gap_record(segment_id, segment, column))

    event_columns = [
        "datetime", "location_segment_id", "location", "measurement", "value", "q1", "q3",
        "iqr", "lower_fence", "upper_fence", "direction",
    ]
    return (
        pd.DataFrame(outlier_summaries),
        pd.DataFrame(outlier_events, columns=event_columns),
        pd.DataFrame(missing_gaps),
    )


def build_profile(frame: pd.DataFrame) -> dict[str, object]:
    """Assemble EDA summaries without changing measurements or removing IQR flags."""
    empty_gaps = _empty_measurement_gaps(frame)
    outlier_summary, outlier_events, missing_gaps = _segment_quality_summaries(frame)
    all_empty = frame[RAW_MEASUREMENT_COLUMNS].isna().all(axis=1)
    overview = {
        "rows": int(len(frame)),
        "start_datetime": frame["datetime"].min().isoformat(),
        "end_datetime": frame["datetime"].max().isoformat(),
        "location_segments": int(frame["location_segment_id"].nunique()),
        "hours_with_no_measurement": int(all_empty.sum()),
        "longest_consecutive_empty_gap_hours": max(empty_gaps, default=0),
        "measurement_columns": RAW_MEASUREMENT_COLUMNS,
        "causal_fill_measurements": MODEL_INPUT_COLUMNS,
        "selected_model_features": SELECTED_FEATURE_COLUMNS,
    }
    return {
        "overview": overview,
        "numeric_summary": _numeric_summary(frame),
        "location_segments": _location_summary(frame),
        "monthly_missing_pct": _monthly_missingness(frame),
        "outlier_summary_by_segment": outlier_summary,
        "outlier_events_by_segment": outlier_events,
        "missing_gap_summary_by_segment": missing_gaps,
        "gap_summary": {
            "empty_gap_count": len(empty_gaps),
            "longest_empty_gap_hours": max(empty_gaps, default=0),
            "empty_gap_hours": sum(empty_gaps),
        },
    }
