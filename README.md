# Citi Bike Net Demand Prediction

Station-level, hour-level net bike demand pipeline for Citi Bike NYC, including weather-merged training data and baseline modeling notebooks.

## Current Status

Completed in this repo:
- Memory-safe rebuild of station-hour training data from raw monthly trip CSVs.
- Weather merge at hourly granularity (`datetime_hour`) into station-hour rows.
- EDA outputs aligned with the report requirements:
  - Hourly demand: `member` vs `casual`
  - Weekday vs weekend hourly demand
  - Top start stations
  - Correlation heatmap including weather features
- Validation and chunked correlation workflow in the merge notebook.

Latest working artifact from notebook flow:
- `data/proceed/micro_mobility_training_data_2023_2023_weather.csv`

---

## Repository Structure

- `merge_trip_weather_v1.0.ipynb`  
  Main end-to-end pipeline notebook (streaming/chunked processing).
- `merge_trip_weather_v1_0_support.py`  
  Support functions used by the notebook (aggregation, merge, validation, correlation).
- `EDA_v.1.0.ipynb`  
  Earlier EDA exploration notebook.
- `XGB_checkpoint_v.1.0.ipynb`  
  Baseline XGBoost training/evaluation notebook.
- `docs/CitiBike_Final_Report.docx.md`  
  Report content and EDA requirements.
- `docs/weather_data_merge_process.md`  
  Weather merge strategy and validation notes.

---

## Data Layout (Expected)

Raw inputs (not tracked in Git):
- `data/citibike/`  
  Monthly Citi Bike trip CSV files (filename prefix `YYYYMM...`).
- `data/weather/open-meteo-2023-2025.csv`  
  Weather CSV with metadata rows (loaded with `skiprows=3` in support code).

Generated outputs:
- `data/proceed/micro_mobility_training_data_<year_start>_<year_end>_weather.csv`
- `data/proceed/merge_trip_weather_v1_0_work/`
  - `monthly_station_hour/*.csv`
  - `eda/cleaning_summary.csv`
  - `eda/hourly_member_casual.csv`
  - `eda/hourly_weekday_weekend.csv`
  - `eda/top_start_stations.csv`
  - `eda/feature_correlation.csv`
  - `monthly_processing_summary.csv`
  - `coords_lookup.csv`

---

## Environment

Recommended:
- Python 3.10+
- Jupyter Notebook / JupyterLab

Core packages:
- `pandas`
- `numpy`
- `matplotlib`
- `ipython`
- `jupyter`

Model notebook extras:
- `xgboost`
- `scikit-learn`

Example install:

```bash
pip install pandas numpy matplotlib ipython jupyter xgboost scikit-learn
```

---

## Run the Main Pipeline

1. Open `merge_trip_weather_v1.0.ipynb`.
2. In the setup cell, set:
   - `PROJECT_ROOT = Path.cwd()`
   - `TARGET_YEARS` (example: `{2023}`)
   - `RESET_OUTPUTS`:
     - `False` to protect existing generated files
     - `True` to rebuild and overwrite generated files
3. Run cells in order.

The notebook will:
- stream trip CSVs in chunks,
- aggregate monthly station-hour net demand,
- generate EDA summaries,
- merge weather features,
- write final dataset CSV,
- validate row counts/nulls,
- compute correlation in chunks and plot heatmap.

---

## Notes on Correlation / Weather Columns

If weather columns do not appear in the final correlation heatmap:
- rerun the notebook import + correlation cells (they now reload the support module),
- ensure `feature_correlation.csv` is not locked by another app (Excel/editor),
- rerun the correlation cell to overwrite the stale file.

---

## Git Tracking Guidance

Recommended to commit:
- notebooks (`*.ipynb`)
- scripts (`*.py`)
- docs (`docs/*`)

Recommended **not** to commit:
- large/generated data under `data/proceed/`
- raw data under `data/citibike/` and `data/weather/`

Add this to `.gitignore` if needed:

```gitignore
data/proceed/
```

---

## Baseline Model Snapshot

From `XGB_checkpoint_v.1.0.ipynb` (reported in notebook output):
- Model: `XGBoost Regressor`
- Chronological split with final 7 days as test
- Reported metrics:
  - RMSE: `3.67`
  - MAE: `0.96`

---

## Next Work (Planned)

- Expand multi-year training window consistently.
- Benchmark additional models vs XGBoost.
- Tighten forecasting-safe weather feature strategy (lagged/forecast-weather inputs).
- Add reproducible dependency lock file and CLI runner.
