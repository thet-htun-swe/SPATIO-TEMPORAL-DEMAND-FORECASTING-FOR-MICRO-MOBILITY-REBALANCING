from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import gc

import numpy as np
import pandas as pd


TRIP_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

WEATHER_RENAME_MAP = {
    "temperature_2m (°C)": "temp_2m",
    "temperature_2m (?C)": "temp_2m",
    "temperature_2m (C)": "temp_2m",
    "relative_humidity_2m (%)": "rh_2m",
    "rain (mm)": "rain_mm",
    "snowfall (cm)": "snow_cm",
    "wind_speed_10m (km/h)": "wind_kmh",
    "precipitation (mm)": "precip_mm",
    "cloud_cover (%)": "cloud_cover",
    "cloud_cover_low (%)": "cloud_low",
    "cloud_cover_mid (%)": "cloud_mid",
    "cloud_cover_high (%)": "cloud_high",
}

WEATHER_COLS = [
    "datetime_hour",
    "temp_2m",
    "rh_2m",
    "rain_mm",
    "snow_cm",
    "wind_kmh",
    "precip_mm",
    "cloud_cover",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
]

FINAL_ORDERED_COLS = [
    "station",
    "date",
    "hour",
    "datetime_hour",
    "is_weekend",
    "net_demand",
    "lat",
    "lng",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "day_sin",
    "day_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "rolling_mean_3h",
    "temp_2m",
    "rh_2m",
    "rain_mm",
    "snow_cm",
    "wind_kmh",
    "precip_mm",
    "cloud_cover",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "is_raining",
    "is_snowing",
]

TRIP_CORR_COLS = [
    "net_demand",
    "lat",
    "lng",
    "hour",
    "is_weekend",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "rolling_mean_3h",
    "temp_2m",
    "rh_2m",
    "rain_mm",
    "snow_cm",
    "wind_kmh",
    "precip_mm",
    "cloud_cover",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "is_raining",
    "is_snowing",
]


@dataclass(frozen=True)
class MergePaths:
    project_root: Path
    trip_root: Path
    weather_path: Path
    output_dir: Path
    work_dir: Path
    monthly_agg_dir: Path
    eda_dir: Path
    output_path: Path
    coords_path: Path
    cleaning_summary_path: Path
    member_hourly_path: Path
    daytype_hourly_path: Path
    top_start_path: Path
    monthly_summary_path: Path
    correlation_path: Path


def build_paths(project_root: Path, target_years: set[int]) -> MergePaths:
    year_label = f"{min(target_years)}_{max(target_years)}"
    output_dir = project_root / "data" / "proceed"
    work_dir = output_dir / "merge_trip_weather_v1_0_work"
    monthly_agg_dir = work_dir / "monthly_station_hour"
    eda_dir = work_dir / "eda"
    return MergePaths(
        project_root=project_root,
        trip_root=project_root / "data" / "citibike",
        weather_path=project_root / "data" / "weather" / "open-meteo-2023-2025.csv",
        output_dir=output_dir,
        work_dir=work_dir,
        monthly_agg_dir=monthly_agg_dir,
        eda_dir=eda_dir,
        output_path=output_dir / f"micro_mobility_training_data_{year_label}_weather.csv",
        coords_path=work_dir / "coords_lookup.csv",
        cleaning_summary_path=eda_dir / "cleaning_summary.csv",
        member_hourly_path=eda_dir / "hourly_member_casual.csv",
        daytype_hourly_path=eda_dir / "hourly_weekday_weekend.csv",
        top_start_path=eda_dir / "top_start_stations.csv",
        monthly_summary_path=work_dir / "monthly_processing_summary.csv",
        correlation_path=eda_dir / "feature_correlation.csv",
    )


def _generated_non_monthly_paths(paths: MergePaths) -> list[Path]:
    return [
        paths.output_path,
        paths.coords_path,
        paths.cleaning_summary_path,
        paths.member_hourly_path,
        paths.daytype_hourly_path,
        paths.top_start_path,
        paths.monthly_summary_path,
        paths.correlation_path,
    ]


def prepare_output_area(paths: MergePaths, reset_outputs: bool = False) -> None:
    for directory in [paths.output_dir, paths.work_dir, paths.monthly_agg_dir, paths.eda_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    existing = [path for path in _generated_non_monthly_paths(paths) if path.exists()]
    existing.extend(sorted(paths.monthly_agg_dir.glob("*.csv")))

    if existing and not reset_outputs:
        existing_text = "\n".join(str(path) for path in existing[:10])
        raise FileExistsError(
            "Notebook-generated outputs already exist. "
            "Set RESET_OUTPUTS = True to overwrite them.\n"
            f"{existing_text}"
        )

    if reset_outputs:
        for path in _generated_non_monthly_paths(paths):
            if path.exists():
                path.unlink()
        for path in paths.monthly_agg_dir.glob("*.csv"):
            path.unlink()


def discover_trip_files(root: Path, years: set[int]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        month_token = path.name[:6]
        if len(month_token) == 6 and month_token.isdigit():
            year = int(month_token[:4])
            if year in years:
                files.append(path)
    return files


def group_trip_files_by_month(files: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[path.name[:6]].append(path)
    return dict(sorted(grouped.items()))


def month_hour_index(month_key: str) -> pd.DatetimeIndex:
    month_start = pd.Timestamp(year=int(month_key[:4]), month=int(month_key[4:]), day=1, hour=0)
    next_month = month_start + pd.offsets.MonthBegin(1)
    return pd.date_range(start=month_start, end=next_month - pd.Timedelta(hours=1), freq="h")


def load_weather_df(weather_path: Path) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_path, skiprows=3)
    weather_df["datetime_hour"] = pd.to_datetime(weather_df["time"], errors="coerce")
    weather_df = weather_df.dropna(subset=["datetime_hour"]).copy()
    weather_df = weather_df.rename(columns=WEATHER_RENAME_MAP)
    missing_cols = [column for column in WEATHER_COLS if column not in weather_df.columns]
    if missing_cols:
        raise KeyError(f"Weather CSV is missing expected columns after normalization: {missing_cols}")
    weather_df = weather_df[WEATHER_COLS].sort_values("datetime_hour").reset_index(drop=True)
    return weather_df


def _append_grouped_counts(
    accumulator: pd.DataFrame | None,
    chunk_counts: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    flush_threshold: int,
) -> pd.DataFrame:
    if accumulator is None or accumulator.empty:
        combined = chunk_counts
    else:
        combined = pd.concat([accumulator, chunk_counts], ignore_index=True)

    if len(combined) >= flush_threshold:
        combined = combined.groupby(group_cols, as_index=False)[value_col].sum()

    return combined


def _update_coord_accumulator(
    accumulator: dict[str, list[float]],
    df: pd.DataFrame,
    station_col: str,
    lat_col: str,
    lng_col: str,
) -> None:
    grouped = (
        df.dropna(subset=[station_col, lat_col, lng_col])
        .groupby(station_col)
        .agg(
            lat_sum=(lat_col, "sum"),
            lng_sum=(lng_col, "sum"),
            coord_count=(lat_col, "size"),
        )
        .reset_index()
    )

    for row in grouped.itertuples(index=False):
        state = accumulator.setdefault(row[0], [0.0, 0.0, 0])
        state[0] += float(row.lat_sum)
        state[1] += float(row.lng_sum)
        state[2] += int(row.coord_count)


def _update_trip_eda_counters(
    hourly_member_counter: dict[tuple[int, str], int],
    hourly_daytype_counter: dict[tuple[int, int], int],
    start_station_counter: dict[str, int],
    clean_chunk: pd.DataFrame,
) -> None:
    hourly_member = clean_chunk.groupby(["hour", "member_casual"]).size()
    for (hour, rider_type), value in hourly_member.items():
        hourly_member_counter[(int(hour), str(rider_type))] += int(value)

    hourly_daytype = clean_chunk.groupby(["hour", "is_weekend"]).size()
    for (hour, is_weekend), value in hourly_daytype.items():
        hourly_daytype_counter[(int(hour), int(is_weekend))] += int(value)

    top_start_counts = clean_chunk.groupby("start_station_name").size()
    for station, value in top_start_counts.items():
        start_station_counter[str(station)] += int(value)


def _counter_dict_to_frame(
    counter: dict[tuple[int, str], int],
    columns: list[str],
    value_name: str,
) -> pd.DataFrame:
    rows = [(*key, value) for key, value in counter.items()]
    return pd.DataFrame(rows, columns=[*columns, value_name])


def run_streamed_trip_aggregation(
    paths: MergePaths,
    target_years: set[int],
    chunk_size: int = 250_000,
    agg_flush_threshold: int = 600_000,
) -> dict[str, object]:
    trip_files = discover_trip_files(paths.trip_root, target_years)
    trip_files_by_month = group_trip_files_by_month(trip_files)

    if not trip_files:
        raise FileNotFoundError(f"No trip CSV files found under {paths.trip_root} for {sorted(target_years)}")

    stats: dict[str, int] = defaultdict(int)
    coords_accum: dict[str, list[float]] = {}
    hourly_member_counter: dict[tuple[int, str], int] = defaultdict(int)
    hourly_daytype_counter: dict[tuple[int, int], int] = defaultdict(int)
    start_station_counter: dict[str, int] = defaultdict(int)
    monthly_summary_records: list[dict[str, object]] = []
    global_date_min: pd.Timestamp | None = None
    global_date_max: pd.Timestamp | None = None
    observed_station_hours_total = 0

    for month_key, month_paths in trip_files_by_month.items():
        print(f"Processing month {month_key} across {len(month_paths)} file(s)")
        month_out: pd.DataFrame | None = None
        month_in: pd.DataFrame | None = None
        month_raw_rows = 0
        month_clean_rows = 0

        for path in month_paths:
            print(f"  reading {path.name}")
            chunk_iter = pd.read_csv(path, usecols=TRIP_COLUMNS, chunksize=chunk_size)

            for chunk in chunk_iter:
                chunk_len = len(chunk)
                month_raw_rows += chunk_len
                stats["raw_rows"] += chunk_len

                chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
                chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")

                invalid_timestamp_mask = chunk[["started_at", "ended_at"]].isna().any(axis=1)
                invalid_timestamp_rows = int(invalid_timestamp_mask.sum())
                stats["invalid_timestamp_rows"] += invalid_timestamp_rows
                if invalid_timestamp_rows:
                    chunk = chunk.loc[~invalid_timestamp_mask].copy()
                else:
                    chunk = chunk.copy()

                chunk["duration_mins"] = (
                    chunk["ended_at"] - chunk["started_at"]
                ).dt.total_seconds() / 60.0

                duration_under_mask = chunk["duration_mins"] < 1
                duration_over_mask = chunk["duration_mins"] > 1_440
                stats["duration_under_1min_rows"] += int(duration_under_mask.sum())
                stats["duration_over_1440min_rows"] += int(duration_over_mask.sum())
                chunk = chunk.loc[~(duration_under_mask | duration_over_mask)].copy()

                missing_end_mask = chunk[["end_station_name", "end_lat", "end_lng"]].isna().any(axis=1)
                stats["missing_end_station_rows"] += int(missing_end_mask.sum())
                if missing_end_mask.any():
                    chunk = chunk.loc[~missing_end_mask].copy()

                missing_start_mask = chunk["start_station_name"].isna()
                stats["missing_start_station_rows"] += int(missing_start_mask.sum())
                if missing_start_mask.any():
                    chunk = chunk.loc[~missing_start_mask].copy()

                if chunk.empty:
                    continue

                month_clean_rows += len(chunk)
                stats["clean_rows"] += len(chunk)

                chunk["hour"] = chunk["started_at"].dt.hour.astype("int8")
                chunk["day_of_week"] = chunk["started_at"].dt.dayofweek.astype("int8")
                chunk["is_weekend"] = (chunk["day_of_week"] >= 5).astype("int8")
                chunk["start_hour_ts"] = chunk["started_at"].dt.floor("h")
                chunk["end_hour_ts"] = chunk["ended_at"].dt.floor("h")

                chunk_min = chunk["started_at"].min()
                chunk_max = chunk["started_at"].max()
                global_date_min = chunk_min if global_date_min is None else min(global_date_min, chunk_min)
                global_date_max = chunk_max if global_date_max is None else max(global_date_max, chunk_max)

                _update_trip_eda_counters(
                    hourly_member_counter,
                    hourly_daytype_counter,
                    start_station_counter,
                    chunk,
                )

                _update_coord_accumulator(coords_accum, chunk, "start_station_name", "start_lat", "start_lng")
                _update_coord_accumulator(coords_accum, chunk, "end_station_name", "end_lat", "end_lng")

                out_chunk = (
                    chunk.groupby(["start_station_name", "start_hour_ts"])
                    .size()
                    .reset_index(name="outflow")
                    .rename(columns={"start_station_name": "station", "start_hour_ts": "datetime_hour"})
                )
                in_chunk = (
                    chunk.groupby(["end_station_name", "end_hour_ts"])
                    .size()
                    .reset_index(name="inflow")
                    .rename(columns={"end_station_name": "station", "end_hour_ts": "datetime_hour"})
                )

                month_out = _append_grouped_counts(
                    month_out,
                    out_chunk,
                    ["station", "datetime_hour"],
                    "outflow",
                    agg_flush_threshold,
                )
                month_in = _append_grouped_counts(
                    month_in,
                    in_chunk,
                    ["station", "datetime_hour"],
                    "inflow",
                    agg_flush_threshold,
                )

                del chunk, out_chunk, in_chunk
                gc.collect()

        if month_out is None:
            month_out = pd.DataFrame(columns=["station", "datetime_hour", "outflow"])
        if month_in is None:
            month_in = pd.DataFrame(columns=["station", "datetime_hour", "inflow"])

        if not month_out.empty:
            month_out = month_out.groupby(["station", "datetime_hour"], as_index=False)["outflow"].sum()
        if not month_in.empty:
            month_in = month_in.groupby(["station", "datetime_hour"], as_index=False)["inflow"].sum()

        month_net = month_out.merge(month_in, on=["station", "datetime_hour"], how="outer").fillna(0)
        month_net["outflow"] = month_net["outflow"].astype("int32")
        month_net["inflow"] = month_net["inflow"].astype("int32")
        month_net["net_demand"] = (month_net["inflow"] - month_net["outflow"]).astype("int32")
        month_net = month_net.sort_values(["station", "datetime_hour"]).reset_index(drop=True)

        month_output_path = paths.monthly_agg_dir / f"station_hour_net_flow_{month_key}.csv"
        month_net.to_csv(month_output_path, index=False)
        observed_station_hours_total += len(month_net)

        monthly_summary_records.append(
            {
                "month_key": month_key,
                "source_file_count": len(month_paths),
                "raw_rows": month_raw_rows,
                "clean_rows": month_clean_rows,
                "observed_station_hour_rows": len(month_net),
            }
        )

        print(
            f"  saved {month_output_path.name} with {len(month_net):,} observed rows "
            f"from {month_clean_rows:,} clean trips"
        )

        del month_out, month_in, month_net
        gc.collect()

    coords_rows = []
    for station, (lat_sum, lng_sum, coord_count) in coords_accum.items():
        if coord_count:
            coords_rows.append(
                {
                    "station": station,
                    "lat": lat_sum / coord_count,
                    "lng": lng_sum / coord_count,
                    "coord_count": coord_count,
                }
            )
    coords_lookup = pd.DataFrame(coords_rows).sort_values("station").reset_index(drop=True)
    coords_lookup.to_csv(paths.coords_path, index=False)

    member_hourly_df = _counter_dict_to_frame(
        hourly_member_counter,
        ["hour", "member_casual"],
        "trip_count",
    ).sort_values(["hour", "member_casual"]).reset_index(drop=True)
    member_hourly_df.to_csv(paths.member_hourly_path, index=False)

    daytype_hourly_df = _counter_dict_to_frame(
        hourly_daytype_counter,
        ["hour", "is_weekend"],
        "trip_count",
    ).sort_values(["hour", "is_weekend"]).reset_index(drop=True)
    daytype_hourly_df["day_type"] = np.where(daytype_hourly_df["is_weekend"] == 1, "weekend", "weekday")
    daytype_hourly_df.to_csv(paths.daytype_hourly_path, index=False)

    top_start_df = (
        pd.DataFrame(
            [{"station": station, "trip_count": count} for station, count in start_station_counter.items()]
        )
        .sort_values("trip_count", ascending=False)
        .reset_index(drop=True)
    )
    top_start_df.to_csv(paths.top_start_path, index=False)

    monthly_summary_df = pd.DataFrame(monthly_summary_records)
    monthly_summary_df.to_csv(paths.monthly_summary_path, index=False)

    cleaning_summary_df = pd.DataFrame(
        [
            {"metric": "trip_file_count", "value": len(trip_files)},
            {"metric": "month_count", "value": len(trip_files_by_month)},
            {"metric": "raw_rows", "value": stats["raw_rows"]},
            {"metric": "invalid_timestamp_rows", "value": stats["invalid_timestamp_rows"]},
            {"metric": "duration_under_1min_rows", "value": stats["duration_under_1min_rows"]},
            {"metric": "duration_over_1440min_rows", "value": stats["duration_over_1440min_rows"]},
            {"metric": "missing_end_station_rows", "value": stats["missing_end_station_rows"]},
            {"metric": "missing_start_station_rows", "value": stats["missing_start_station_rows"]},
            {"metric": "clean_rows", "value": stats["clean_rows"]},
            {"metric": "station_count", "value": len(coords_lookup)},
            {"metric": "observed_station_hour_rows", "value": observed_station_hours_total},
            {"metric": "started_at_min", "value": str(global_date_min)},
            {"metric": "started_at_max", "value": str(global_date_max)},
        ]
    )
    cleaning_summary_df.to_csv(paths.cleaning_summary_path, index=False)

    return {
        "trip_files": trip_files,
        "trip_files_by_month": trip_files_by_month,
        "global_date_min": global_date_min,
        "global_date_max": global_date_max,
        "cleaning_summary": cleaning_summary_df,
        "coords_lookup": coords_lookup,
        "monthly_summary": monthly_summary_df,
    }


def build_final_dataset(
    paths: MergePaths,
    station_batch_size: int = 64,
) -> dict[str, object]:
    coords_lookup = pd.read_csv(paths.coords_path).sort_values("station").reset_index(drop=True)
    stations = coords_lookup["station"].astype(str).to_numpy()
    lat_values = coords_lookup["lat"].astype("float32").to_numpy()
    lng_values = coords_lookup["lng"].astype("float32").to_numpy()

    monthly_files = sorted(paths.monthly_agg_dir.glob("station_hour_net_flow_*.csv"))
    weather_df = load_weather_df(paths.weather_path).set_index("datetime_hour").sort_index()

    history = np.zeros((24, len(stations)), dtype=np.float32)
    write_header = True
    total_rows_written = 0
    month_write_summary: list[dict[str, object]] = []

    for month_path in monthly_files:
        month_key = month_path.stem.rsplit("_", 1)[-1]
        hour_index = month_hour_index(month_key)
        hour_count = len(hour_index)

        month_df = pd.read_csv(month_path, parse_dates=["datetime_hour"])
        month_pivot = (
            month_df.pivot_table(
                index="datetime_hour",
                columns="station",
                values="net_demand",
                aggfunc="sum",
            )
            .reindex(hour_index)
            .reindex(columns=stations, fill_value=0.0)
            .fillna(0.0)
        )
        month_matrix = month_pivot.to_numpy(dtype=np.float32)
        extended = np.vstack([history, month_matrix])

        lag_1 = extended[23 : 23 + hour_count]
        lag_2 = extended[22 : 22 + hour_count]
        lag_3 = extended[21 : 21 + hour_count]
        lag_24 = extended[:hour_count]
        rolling_mean_3h = (lag_1 + lag_2 + lag_3) / 3.0

        weather_month = weather_df.reindex(hour_index).copy()
        weather_month["is_raining"] = (weather_month["precip_mm"].fillna(0) > 0).astype("int8")
        weather_month["is_snowing"] = (weather_month["snow_cm"].fillna(0) > 0).astype("int8")

        date_values = hour_index.normalize().to_numpy()
        datetime_values = hour_index.to_numpy()
        hour_values = hour_index.hour.to_numpy(dtype=np.int8)
        day_of_week_values = hour_index.dayofweek.to_numpy(dtype=np.int8)
        is_weekend_values = (day_of_week_values >= 5).astype(np.int8)
        hour_sin_values = np.sin(2 * np.pi * hour_values / 24).astype(np.float32)
        hour_cos_values = np.cos(2 * np.pi * hour_values / 24).astype(np.float32)
        day_sin_values = np.sin(2 * np.pi * day_of_week_values / 7).astype(np.float32)
        day_cos_values = np.cos(2 * np.pi * day_of_week_values / 7).astype(np.float32)

        weather_arrays = {
            column: weather_month[column].to_numpy()
            for column in [
                "temp_2m",
                "rh_2m",
                "rain_mm",
                "snow_cm",
                "wind_kmh",
                "precip_mm",
                "cloud_cover",
                "cloud_low",
                "cloud_mid",
                "cloud_high",
                "is_raining",
                "is_snowing",
            ]
        }

        month_rows_written = 0
        for start_idx in range(0, len(stations), station_batch_size):
            end_idx = min(start_idx + station_batch_size, len(stations))
            station_slice = slice(start_idx, end_idx)
            batch_station_count = end_idx - start_idx

            batch_df = pd.DataFrame(
                {
                    "station": np.repeat(stations[station_slice], hour_count),
                    "date": np.tile(date_values, batch_station_count),
                    "hour": np.tile(hour_values, batch_station_count),
                    "datetime_hour": np.tile(datetime_values, batch_station_count),
                    "is_weekend": np.tile(is_weekend_values, batch_station_count),
                    "net_demand": month_matrix[:, station_slice].T.reshape(-1),
                    "lat": np.repeat(lat_values[station_slice], hour_count),
                    "lng": np.repeat(lng_values[station_slice], hour_count),
                    "hour_sin": np.tile(hour_sin_values, batch_station_count),
                    "hour_cos": np.tile(hour_cos_values, batch_station_count),
                    "day_of_week": np.tile(day_of_week_values, batch_station_count),
                    "day_sin": np.tile(day_sin_values, batch_station_count),
                    "day_cos": np.tile(day_cos_values, batch_station_count),
                    "lag_1h": lag_1[:, station_slice].T.reshape(-1),
                    "lag_2h": lag_2[:, station_slice].T.reshape(-1),
                    "lag_3h": lag_3[:, station_slice].T.reshape(-1),
                    "lag_24h": lag_24[:, station_slice].T.reshape(-1),
                    "rolling_mean_3h": rolling_mean_3h[:, station_slice].T.reshape(-1),
                }
            )

            for column, values in weather_arrays.items():
                batch_df[column] = np.tile(values, batch_station_count)

            batch_df = batch_df[FINAL_ORDERED_COLS]
            batch_df.to_csv(paths.output_path, mode="a", header=write_header, index=False)
            write_header = False

            batch_row_count = len(batch_df)
            total_rows_written += batch_row_count
            month_rows_written += batch_row_count

        history = extended[-24:].copy()
        month_write_summary.append(
            {
                "month_key": month_key,
                "hour_count": hour_count,
                "rows_written": month_rows_written,
            }
        )
        print(f"Wrote {month_rows_written:,} rows for month {month_key}")

        del month_df, month_pivot, month_matrix, extended, lag_1, lag_2, lag_3, lag_24, rolling_mean_3h
        gc.collect()

    return {
        "station_count": len(stations),
        "month_count": len(monthly_files),
        "rows_written": total_rows_written,
        "month_write_summary": pd.DataFrame(month_write_summary),
    }


def validate_final_output(
    paths: MergePaths,
    chunksize: int = 200_000,
) -> dict[str, object]:
    coords_lookup = pd.read_csv(paths.coords_path)
    monthly_files = sorted(paths.monthly_agg_dir.glob("station_hour_net_flow_*.csv"))
    station_count = len(coords_lookup)
    expected_rows = station_count * sum(len(month_hour_index(path.stem.rsplit("_", 1)[-1])) for path in monthly_files)

    actual_rows = 0
    weather_cols = ["temp_2m", "rh_2m", "precip_mm", "snow_cm", "wind_kmh", "cloud_cover"]
    weather_null_counts = dict.fromkeys(weather_cols, 0)

    for chunk in pd.read_csv(paths.output_path, usecols=["station", *weather_cols], chunksize=chunksize):
        actual_rows += len(chunk)
        for column in weather_cols:
            weather_null_counts[column] += int(chunk[column].isna().sum())

    weather_null_pct = {
        column: round((count / actual_rows) * 100, 4) if actual_rows else 0.0
        for column, count in weather_null_counts.items()
    }

    return {
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "rows_match_expected": expected_rows == actual_rows,
        "station_count": station_count,
        "weather_null_pct": weather_null_pct,
    }


def compute_chunked_correlation(
    csv_path: Path,
    numeric_cols: list[str],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    n_rows = 0
    col_count = len(numeric_cols)
    sum_x = np.zeros(col_count, dtype=np.float64)
    sum_x2 = np.zeros(col_count, dtype=np.float64)
    sum_xy = np.zeros((col_count, col_count), dtype=np.float64)

    for chunk in pd.read_csv(csv_path, usecols=numeric_cols, chunksize=chunksize):
        values = chunk.astype(np.float64).to_numpy()
        n_rows += len(values)
        sum_x += values.sum(axis=0)
        sum_x2 += np.square(values).sum(axis=0)
        sum_xy += values.T @ values

    means = sum_x / n_rows
    variances = (sum_x2 / n_rows) - np.square(means)
    variances = np.clip(variances, a_min=0.0, a_max=None)
    std = np.sqrt(variances)

    covariance = (sum_xy / n_rows) - np.outer(means, means)
    denom = np.outer(std, std)
    correlation = np.divide(
        covariance,
        denom,
        out=np.zeros_like(covariance),
        where=denom != 0,
    )

    return pd.DataFrame(correlation, index=numeric_cols, columns=numeric_cols)


def load_eda_outputs(paths: MergePaths) -> dict[str, pd.DataFrame]:
    return {
        "cleaning_summary": pd.read_csv(paths.cleaning_summary_path),
        "member_hourly": pd.read_csv(paths.member_hourly_path),
        "daytype_hourly": pd.read_csv(paths.daytype_hourly_path),
        "top_start": pd.read_csv(paths.top_start_path),
        "monthly_summary": pd.read_csv(paths.monthly_summary_path),
    }
