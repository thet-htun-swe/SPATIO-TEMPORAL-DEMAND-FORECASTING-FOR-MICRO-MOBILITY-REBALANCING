# Weather Data Merge Process

## Purpose

This document defines the recommended process for merging weather data into the Citi Bike station-hour training dataset. The goal is to keep the merge logic explicit, reproducible, and safe for later model iterations, especially when extending the trip dataset to multiple years.

## Current Inputs

### Trip training dataset

Source:
- `data/proceed/micro_mobility_training_data_2025.csv`

Current grain:
- One row per `station x date x hour`

Core columns already present:
- `station`
- `date`
- `hour`
- `net_demand`
- `lat`, `lng`
- temporal encodings
- lag features
- rolling features

### Weather dataset

Source:
- `data/weather/open-meteo-2023-2025.csv`

Weather grain:
- One row per hour

Observed structure:
- Row 0: metadata headers
- Row 1: metadata values
- Row 2: blank
- Row 3: actual weather header
- Data rows begin after row 3

Required parsing note:
- Read with `skiprows=3`

Weather time coverage:
- `2023-01-01 00:00:00` to `2025-12-31 23:00:00`

2025 weather columns:
- `time`
- `temperature_2m (°C)`
- `relative_humidity_2m (%)`
- `rain (mm)`
- `snowfall (cm)`
- `wind_speed_10m (km/h)`
- `precipitation (mm)`
- `cloud_cover (%)`
- `cloud_cover_low (%)`
- `cloud_cover_mid (%)`
- `cloud_cover_high (%)`

## Merge Strategy

### Merge grain

The correct merge grain is:

- trip data at `station x date x hour`
- weather data at `datetime hour`

Since weather is city-level, the same hourly weather row should be replicated across all stations for that hour.

### Join key

Create a single timestamp key on both datasets:

- Trip side: `datetime_hour = to_datetime(date) + hour`
- Weather side: parse `time` as datetime and rename to `datetime_hour`

Recommended merge:
- `left join` from trip data to weather data on `datetime_hour`

This preserves all trip rows and attaches one hourly weather profile to each station-hour row.

## Recommended Workflow

1. Load trip training data.
2. Parse `date` as datetime.
3. Create `datetime_hour` from `date` and `hour`.
4. Load weather CSV using `skiprows=3`.
5. Parse `time` as datetime.
6. Filter weather to the same year range as the trip dataset.
7. Rename weather columns to model-friendly names.
8. Merge weather into the trip training dataset using `datetime_hour`.
9. Validate row count, uniqueness, and null coverage.
10. Save a new merged training dataset version rather than overwriting the current file.

## Recommended Column Renaming

Use simplified names after loading weather:

- `temperature_2m (°C)` -> `temp_2m`
- `relative_humidity_2m (%)` -> `rh_2m`
- `rain (mm)` -> `rain_mm`
- `snowfall (cm)` -> `snow_cm`
- `wind_speed_10m (km/h)` -> `wind_kmh`
- `precipitation (mm)` -> `precip_mm`
- `cloud_cover (%)` -> `cloud_cover`
- `cloud_cover_low (%)` -> `cloud_low`
- `cloud_cover_mid (%)` -> `cloud_mid`
- `cloud_cover_high (%)` -> `cloud_high`

## Recommended Weather Features

### Strong first-pass features

- `temp_2m`
- `rh_2m`
- `precip_mm`
- `snow_cm`
- `wind_kmh`
- `cloud_cover`

### Optional detailed features

- `rain_mm`
- `cloud_low`
- `cloud_mid`
- `cloud_high`

These may help, but they can also be redundant. Start simple before expanding.

### Optional derived features

- `is_raining = precip_mm > 0`
- `is_snowing = snow_cm > 0`
- `heavy_precip = precip_mm > threshold`
- `high_wind = wind_kmh > threshold`
- `bad_weather = heavy_precip or is_snowing or high_wind`

Tree-based models usually handle a mix of raw continuous features and simple flags well.

## Leakage and Forecasting Rules

This is the main modeling caution.

### Safe for analysis / nowcasting

Using weather observed at the same hour as the target hour is acceptable if the use case is:
- retrospective analysis
- same-hour estimation
- nowcasting

### Not strictly safe for future forecasting

Using actual weather from the target hour is not leakage-safe if the task is:
- forecasting future hours ahead
- generating operational predictions before the hour happens

For strict forecasting, prefer one of these:
- lagged weather features such as `weather_t-1h`, `weather_t-2h`, `weather_t-3h`
- rolling past-weather summaries
- forecast weather available at prediction time

## Validation Checklist

After merging, confirm all of the following:

- Row count remains unchanged from the trip dataset.
- For 2025, the merged row count remains `4,266,120` if starting from the current training file.
- No duplicate rows exist for the key `station + date + hour`.
- Weather coverage is complete or near-complete for the target date range.
- `datetime_hour` aligns to New York local time.
- No unexpected shifts exist between trip-hour and weather-hour timestamps.

Recommended checks:

- count rows before and after merge
- count distinct `station, date, hour` before and after merge
- inspect weather null percentages per column
- inspect a few known timestamps manually

## Output Recommendation

Do not overwrite the existing training file.

Create a new versioned output such as:

- `data/proceed/micro_mobility_training_data_2025_weather.csv`

If expanding to multiple years, use a year-range naming pattern such as:

- `data/proceed/micro_mobility_training_data_2023_2025_weather.csv`

## Multi-Year Extension Plan

When the past 3 years of trip data are ready:

1. Rebuild the trip training dataset at the same `station x date x hour` grain for all years.
2. Ensure station-hour continuity and lag generation are computed across the full chronological sequence.
3. Filter weather to the same full date range.
4. Merge weather using the same `datetime_hour` key.
5. Recompute train/test split chronologically after the merge.

Important note:
- If training spans multiple years, verify daylight saving transitions explicitly.
- NYC-local timestamps should remain consistent between trip data and Open-Meteo data.

## Final Recommendation

For the next iteration, the cleanest path is:

- keep the existing 2025 station-hour training table as the base
- merge hourly weather on `datetime_hour`
- start with a compact weather feature set
- document whether the experiment is nowcasting or forecasting
- if forecasting, convert weather inputs to lagged or forecast-based features

This keeps the pipeline defensible and avoids ambiguity later when benchmarking new models.
