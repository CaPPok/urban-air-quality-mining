"""Combine daily OpenAQ CMT8 archives into a single two-year hourly-wide CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
from pathlib import Path


PARAMETERS = {
    "pm1": ("pm1_ug_m3", "sensor_id_pm1"),
    "pm25": ("pm2_5_ug_m3", "sensor_id_pm2_5"),
    "pm10": ("pm10_ug_m3", "sensor_id_pm10"),
    "relativehumidity": ("relative_humidity_pct", "sensor_id_relativehumidity"),
    "temperature": ("temperature_c", "sensor_id_temperature"),
    "um003": ("pm0_3_count_particles_cm3", "sensor_id_um003"),
}
BASE_COLUMNS = ["datetime", "location_id", "location", "latitude", "longitude"]


def merge(input_dir: Path, output_path: Path) -> tuple[int, int]:
    paths = sorted(input_dir.glob("location-*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No OpenAQ daily .csv.gz files found in {input_dir}")

    measurements: dict[tuple[dt.datetime, str], str] = {}
    sensor_ids: dict[str, str] = {}
    metadata_by_timestamp: dict[dt.datetime, tuple[str, str, str, str]] = {}
    timestamps: list[dt.datetime] = []

    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                parameter = row["parameter"]
                if parameter not in PARAMETERS:
                    raise ValueError(f"Unexpected parameter {parameter!r} in {path.name}")

                stamp = dt.datetime.fromisoformat(row["datetime"])
                key = (stamp, parameter)
                if key in measurements:
                    raise ValueError(f"Duplicate {parameter} reading at {stamp.isoformat()}")
                measurements[key] = row["value"]
                timestamps.append(stamp)

                sensor_id = row["sensors_id"]
                previous_sensor_id = sensor_ids.setdefault(parameter, sensor_id)
                if previous_sensor_id != sensor_id:
                    raise ValueError(f"Multiple sensor IDs found for parameter {parameter!r}")

                metadata = (
                    row["location_id"],
                    row["location"],
                    row["lat"],
                    row["lon"],
                )
                previous_metadata = metadata_by_timestamp.setdefault(stamp, metadata)
                if previous_metadata != metadata:
                    raise ValueError(f"Location metadata disagrees at {stamp.isoformat()}")

    start, end = min(timestamps), max(timestamps)
    if start.minute or start.second or end.minute or end.second:
        raise ValueError("Expected hourly timestamps aligned to the start of an hour")

    header = BASE_COLUMNS[:]
    for parameter in PARAMETERS:
        value_column, sensor_column = PARAMETERS[parameter]
        header.extend((value_column, sensor_column))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    missing_hour_count = 0
    current_metadata = ("", "", "", "")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        stamp = start
        while stamp <= end:
            # Location metadata is carried forward across empty hours; measured values remain blank.
            current_metadata = metadata_by_timestamp.get(stamp, current_metadata)
            result: dict[str, str] = {
                "datetime": stamp.isoformat(),
                "location_id": current_metadata[0],
                "location": current_metadata[1],
                "latitude": current_metadata[2],
                "longitude": current_metadata[3],
            }
            has_any_reading = False
            for parameter, (value_column, sensor_column) in PARAMETERS.items():
                result[value_column] = measurements.get((stamp, parameter), "")
                result[sensor_column] = sensor_ids.get(parameter, "")
                has_any_reading |= bool(result[value_column])
            writer.writerow(result)
            row_count += 1
            missing_hour_count += int(not has_any_reading)
            stamp += dt.timedelta(hours=1)

    return row_count, missing_hour_count


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "raw" / "openaq_cmt8_2y",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "openaq_cmt8_2y_hourly.csv",
    )
    args = parser.parse_args()
    row_count, missing_hour_count = merge(args.input_dir, args.output)
    print(f"Wrote {row_count:,} hourly rows to {args.output}")
    print(f"Hours with no readings: {missing_hour_count:,}; no values were imputed.")


if __name__ == "__main__":
    main()
