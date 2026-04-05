# Model Training Notebooks

Primary notebook:
- `train_xgb_weather_v2.ipynb`

What it does:
- loads merged `weather_v2` training CSV in chunks,
- runs pre-training schema/null/date checks,
- applies chronological split (last 7 days holdout),
- trains XGBoost,
- reports MAE/RMSE + hourly MAE,
- saves model and metrics into `artifacts/model_training/...`.

Notes:
- No YAML dependency.
- Feature list is defined directly in the notebook.
- Input file is auto-selected from:
  - `data/proceed/micro_mobility_training_data_*_weather_v2.csv`
