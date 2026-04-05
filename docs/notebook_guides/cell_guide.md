# Notebook Cell Guide

## 1) `notebooks/data_preparation/build_trip_weather_eda_v2.ipynb`

### Cell 0 (Markdown)
- Notebook purpose and end-to-end scope.
- States that trip EDA cleaning + weather merge are done in one pipeline.

### Cell 1 (Imports)
- Imports core libraries:
  - `pandas`, `numpy`, `matplotlib`
  - utility modules (`Path`, `dataclass`, `gc`, etc.).

### Cell 2 (Config + Paths)
- Defines pipeline configuration:
  - `TARGET_YEARS`, `CHUNK_SIZE`, `RESET_OUTPUTS`
  - ghost keyword list and low-activity threshold.
- Creates output/work directory structure.
- Defines output files (final dataset + EDA outputs + metadata paths).
- Handles reset cleanup when `RESET_OUTPUTS=True`.

### Cell 3 (Helpers + File Discovery)
- Defines:
  - trip-file discovery logic from `data/citibike`
  - weather loader + column normalization from `data/weather`.
- Discovers trip CSVs for selected years and validates presence.

### Cell 4 (Pass 1: Station Filtering Prep)
- Runs initial pass over trip data to build station removal sets:
  - applies core cleaning (timestamps, duration, missing end station/coords)
  - detects ghost/admin stations by keyword
  - computes low-activity start stations (`< LOW_ACTIVITY_THRESHOLD`)
  - combines both sets into `stations_to_remove`.
- Saves removed stations list.

### Cell 5 (Pass 2: Aggregation + EDA Counters)
- Re-reads trip files and applies all cleaning + station removal.
- Computes EDA counters:
  - hourly member/casual
  - hourly weekday/weekend
  - top start/end stations.
- Builds temporary monthly departure/arrival aggregates for net-flow assembly.
- Collects coordinate statistics for station lat/lng lookup.

### Cell 6 (Finalize Monthly + Save EDA Files)
- Reduces temporary monthly flow files into final monthly station-hour net-flow files.
- Writes:
  - monthly processing summary
  - coordinate lookup
  - EDA CSV outputs (`cleaning_summary`, hourly summaries, top stations).

### Cell 7 (Feature Engineering + Weather Merge)
- Loads weather data.
- Builds final station-hour feature table month by month:
  - full hourly index per month
  - lag features (`lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`)
  - rolling mean (`rolling_mean_3h`)
  - temporal encodings (`hour/day sin-cos`, `is_weekend`)
  - weather features + rain/snow flags.
- Writes final merged training CSV:
  - `data/proceed/micro_mobility_training_data_<year_start>_<year_end>_weather_v2.csv`.

### Cell 8 (Validation Sample + EDA Charts)
- Displays head/tail sample of final output.
- Plots the main EDA visuals:
  - member vs casual hourly demand
  - weekday vs weekend hourly demand
  - top start stations.

---

## 2) `notebooks/model_training/train_xgb_v1_nowcasting.ipynb`

### Cell 0 (Markdown)
- Training notebook purpose and workflow summary.

### Cell 1 (Imports + Paths)
- Imports ML/data libraries.
- Defines `PROJECT_ROOT`, input data path, and artifact output folder.

### Cell 2 (Runtime Controls)
- Sets chunk size, holdout days, row caps, random seed.

### Cell 3 (Feature Definition)
- Defines target, date/station fields, and selected numeric features.

### Cell 4 (Pass 1 Date Range)
- Scans file for min/max date and computes chronological split cutoff.

### Cell 5 (Data Quality Check)
- Computes row count and null percentage for selected columns.

### Cell 6 (Station Encoding + Build Train/Test)
- Builds station-to-id mapping.
- Loads data in chunks and splits by cutoff date.
- Applies row caps for memory-safe training sets.

### Cell 7 (Model Training)
- Trains `XGBRegressor` with baseline parameters.

### Cell 8 (Evaluation)
- Computes MAE/RMSE.
- Generates hourly MAE and feature-importance plots.

### Cell 9 (Artifact Saving)
- Saves model, metrics JSON, hourly MAE CSV, and feature metadata.

### Cell 10 (Markdown)
- Notes on memory tuning and forecast-safe next step.
