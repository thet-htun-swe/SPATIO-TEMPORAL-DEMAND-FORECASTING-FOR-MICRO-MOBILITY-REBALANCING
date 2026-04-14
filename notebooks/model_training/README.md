# Model Training Notebooks

Primary notebook:
- `train_xgb_weather_v2.ipynb`

Additional comparison notebooks (2025 dataset flow):
- `train_mlp_v1.ipynb`
- `train_arimax_v1.ipynb`
- `train_mlp_tuned_v1.ipynb`
- `train_arimax_tuned_v1.ipynb`
- `train_arimax_tuned_safe_v1.ipynb`

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

For `v1` comparison notebooks:
- Input dataset is fixed to `data/proceed/micro_mobility_training_data_2025.csv`.
- Uses the same chronological holdout logic (last 7 days).

For tuned notebooks:
- `train_mlp_tuned_v1.ipynb` runs randomized hyperparameter search with a chronological validation window.
- `train_arimax_tuned_v1.ipynb` runs compact ARIMAX grid search on representative stations, then evaluates best config on a larger station subset.
- `train_arimax_tuned_safe_v1.ipynb` is a leakage-safe ARIMAX variant that uses only calendar exogenous features known at forecast time.
