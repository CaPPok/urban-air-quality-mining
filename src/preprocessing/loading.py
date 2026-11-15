"""Load, validate and regularize hourly OpenAQ data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import METADATA_COLUMNS, RAW_MEASUREMENT_COLUMNS, REQUIRED_COLUMNS


def _location_segment_ids(frame: pd.DataFrame) -> pd.Series:
    """Assign a new segment whenever the source location metadata changes."""
    signature_columns = ["location", "latitude", "longitude"]
    signatures = frame[signature_columns].astype("string").fillna("<unknown>")
    changed = signatures.ne(signatures.shift()).any(axis=1)
    if len(changed):
        changed.iloc[0] = True
    return changed.cumsum().sub(1).astype("int64")


def load_hourly_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load, validate, and regularize the hourly CSV without filling measurements."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    if frame["datetime"].isna().any():
        bad_count = int(frame["datetime"].isna().sum())
        raise ValueError(f"Found {bad_count} invalid timestamps")
    frame["datetime"] = frame["datetime"].dt.tz_convert("Asia/Ho_Chi_Minh")

    duplicate_count = int(frame["datetime"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate hourly timestamps")

    source_rows = len(frame)
    frame = frame.sort_values("datetime").set_index("datetime")
    for column in RAW_MEASUREMENT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Reindex to a complete hourly grid. Measurement gaps remain NaN.
    full_index = pd.date_range(start=frame.index.min(), end=frame.index.max(), freq="h")
    missing_grid_hours = len(full_index.difference(frame.index))
    frame = frame.reindex(full_index)
    frame.index.name = "datetime"
    frame = frame.reset_index()

    # Metadata is descriptive, so carry it through empty hours. Never carry measurements.
    for column in METADATA_COLUMNS:
        frame[column] = frame[column].ffill()
    for column in frame.columns:
        if column.startswith("sensor_id_"):
            frame[column] = frame[column].ffill()

    invalid_counts: dict[str, int] = {}
    for column in ["pm1_ug_m3", "pm2_5_ug_m3", "pm10_ug_m3", "pm0_3_count_particles_cm3"]:
        invalid = frame[column].lt(0)
        invalid_counts[column] = int(invalid.sum())
        frame.loc[invalid, column] = np.nan

    invalid_rh = ~frame["relative_humidity_pct"].between(0, 100) & frame["relative_humidity_pct"].notna()
    invalid_counts["relative_humidity_pct"] = int(invalid_rh.sum())
    frame.loc[invalid_rh, "relative_humidity_pct"] = np.nan

    frame["location_segment_id"] = _location_segment_ids(frame)
    measurement_empty = frame[RAW_MEASUREMENT_COLUMNS].isna().all(axis=1)
    report: dict[str, object] = {
        "source_file": str(path),
        "source_rows": int(source_rows),
        "regularized_rows": int(len(frame)),
        "hours_added_to_grid": int(missing_grid_hours),
        "duplicate_timestamps": duplicate_count,
        "start_datetime": frame["datetime"].min().isoformat(),
        "end_datetime": frame["datetime"].max().isoformat(),
        "location_segment_count": int(frame["location_segment_id"].nunique()),
        "hours_without_any_measurement": int(measurement_empty.sum()),
        "invalid_values_replaced_with_missing": invalid_counts,
    }
    return frame, report
