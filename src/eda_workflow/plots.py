"""Dataset and feature-correlation plots for EDA presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing.schema import RAW_MEASUREMENT_COLUMNS, SELECTED_FEATURE_COLUMNS
from .profile import _monthly_missingness


def _plot_monthly_missingness(frame: pd.DataFrame, figures_dir: Path) -> Path:
    monthly = _monthly_missingness(frame)
    fig, ax = plt.subplots(figsize=(max(11, len(monthly) * 0.42), 4.8))
    image = ax.imshow(monthly.T, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(RAW_MEASUREMENT_COLUMNS)), [column.replace("_", " ") for column in RAW_MEASUREMENT_COLUMNS])
    tick_step = max(1, len(monthly) // 12)
    ticks = list(range(0, len(monthly), tick_step))
    ax.set_xticks(ticks, [monthly.index[i] for i in ticks], rotation=45, ha="right")
    ax.set_title("Missing measurements by month")
    ax.set_xlabel("Month")
    fig.colorbar(image, ax=ax, label="Missing hours (%)")
    fig.tight_layout()
    output_path = figures_dir / "monthly_missingness.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_pm25_timeline(frame: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for segment_id, group in frame.groupby("location_segment_id", sort=True):
        label = str(group["location"].iloc[0])
        ax.plot(group["datetime"], group["pm2_5_ug_m3"], linewidth=0.55, label=f"Segment {segment_id}: {label}")
    ax.set_title("Hourly PM2.5 observations")
    ax.set_xlabel("Local time (Asia/Ho_Chi_Minh)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path = figures_dir / "pm25_timeline.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_pm25_by_location(frame: pd.DataFrame, figures_dir: Path) -> Path:
    box_values, box_labels = [], []
    for _, group in frame.groupby("location_segment_id", sort=True):
        values = group["pm2_5_ug_m3"].dropna().to_numpy()
        if len(values):
            box_values.append(values)
            box_labels.append(f"{group['location'].iloc[0]}\n(n={len(values):,})")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(box_values, showfliers=False)
    ax.set_xticks(range(1, len(box_labels) + 1), box_labels)
    ax.set_title("PM2.5 distribution by location segment")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    output_path = figures_dir / "pm25_by_location.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_pm25_diurnal_profile(frame: pd.DataFrame, figures_dir: Path) -> Path:
    diurnal = frame.assign(hour=frame["datetime"].dt.hour).groupby(["location_segment_id", "hour"])["pm2_5_ug_m3"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for segment_id in sorted(frame["location_segment_id"].unique()):
        if segment_id in diurnal.index.get_level_values(0):
            series = diurnal.loc[segment_id]
            ax.plot(series.index, series.values, marker="o", markersize=2.5, linewidth=1.2, label=f"Segment {segment_id}")
    ax.set_title("Average PM2.5 by hour of day")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    output_path = figures_dir / "pm25_diurnal_profile.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_raw_correlation(frame: pd.DataFrame, figures_dir: Path) -> Path:
    correlation_columns = [
        "pm1_ug_m3",
        "pm2_5_ug_m3",
        "relative_humidity_pct",
        "temperature_c",
        "pm0_3_count_particles_cm3",
    ]
    correlation = frame[correlation_columns].corr(min_periods=24)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    labels = [column.replace("_", " ") for column in correlation_columns]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson correlation (pairwise observations)")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    output_path = figures_dir / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_eda(frame: pd.DataFrame, figures_dir: str | Path) -> dict[str, Path]:
    """Create the core figures for missingness, PM2.5, seasonality and raw correlations."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "monthly_missingness": _plot_monthly_missingness(frame, output_dir),
        "pm25_timeline": _plot_pm25_timeline(frame, output_dir),
        "pm25_by_location": _plot_pm25_by_location(frame, output_dir),
        "pm25_diurnal_profile": _plot_pm25_diurnal_profile(frame, output_dir),
        "correlation_heatmap": _plot_raw_correlation(frame, output_dir),
    }


def plot_feature_target_correlations(
    correlations: pd.DataFrame,
    figures_dir: str | Path,
) -> Path:
    """Plot train-only candidate correlations as descriptive evidence, not importance."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ordered = correlations.sort_values("pearson_with_target", na_position="first")
    colors = [
        "#2878B5" if selected else "#B8C2CC"
        for selected in ordered["selected_for_default_model"]
    ]
    labels = ordered["feature"].str.replace("_", " ", regex=False)

    fig, ax = plt.subplots(figsize=(11, max(5, len(ordered) * 0.34)))
    ax.barh(labels, ordered["pearson_with_target"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with next-hour PM2.5")
    ax.set_title("Candidate features: train-only correlation (descriptive)")
    ax.text(
        0,
        -0.13,
        "Blue = current baseline feature; correlation is not feature importance or proof of model value.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    output_path = figures_dir / "feature_target_correlations_train.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    figures_dir: str | Path,
) -> Path:
    """Plot train-only correlations among candidate inputs; mark baseline features."""
    labels_by_feature = {
        "pm2_5_ug_m3_input_causal": "PM2.5 current",
        "pm2_5_lag_1h": "PM2.5 lag 1h",
        "pm2_5_lag_3h": "PM2.5 lag 3h",
        "pm2_5_lag_24h": "PM2.5 lag 24h",
        "pm2_5_rolling24h_mean": "PM2.5 roll 24h",
        "pm1_ug_m3_input_causal": "PM1 current",
        "pm1_ug_m3_lag_1h": "PM1 lag 1h",
        "relative_humidity_pct_input_causal": "Humidity current",
        "relative_humidity_pct_lag_1h": "Humidity lag 1h",
        "temperature_c_input_causal": "Temperature current",
        "temperature_c_lag_1h": "Temperature lag 1h",
        "pm0_3_count_particles_cm3_input_causal": "PM0.3 current",
        "pm0_3_count_particles_cm3_lag_1h": "PM0.3 lag 1h",
        "hour_sin": "Hour sin",
        "hour_cos": "Hour cos",
        "day_of_week_sin": "Weekday sin",
        "day_of_week_cos": "Weekday cos",
        "month_sin": "Month sin",
        "month_cos": "Month cos",
    }
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    features = list(correlation_matrix.index)
    labels = [
        f"{labels_by_feature.get(feature, feature)} *" if feature in SELECTED_FEATURE_COLUMNS
        else labels_by_feature.get(feature, feature)
        for feature in features
    ]
    size = max(11, len(features) * 0.72)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), labels)

    for tick, feature in zip(ax.get_xticklabels(), features):
        if feature in SELECTED_FEATURE_COLUMNS:
            tick.set_color("#176b3a")
            tick.set_fontweight("bold")
    for tick, feature in zip(ax.get_yticklabels(), features):
        if feature in SELECTED_FEATURE_COLUMNS:
            tick.set_color("#176b3a")
            tick.set_fontweight("bold")

    for row in range(len(features)):
        for column in range(len(features)):
            value = correlation_matrix.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(column, row, label, ha="center", va="center", fontsize=6.5)

    ax.set_title("Candidate feature correlations on training data\n* marks a baseline feature")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.text(
        0.5,
        0.015,
        "Strongly correlated features may contain overlapping information; correlation alone does not determine model value.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = figures_dir / "feature_correlation_matrix_train.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
