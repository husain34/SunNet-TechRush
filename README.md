# SunNet-TechRush
A machine learning model to predict solar energy generation using historical weather data. Incorporates features like temperature, cloud cover, and seasonal patterns for accurate forecasting. Built by Team SunNet for TechRush Hackathon, PISB focused on sustainable energy solutions.

## 🧪 Model Experiments

The following scripts are located in `src/experiments/`:

| File Name                         | Description                                      |
|----------------------------------|--------------------------------------------------|
| `kfold_xgboost.py`               | XGBoost model with K-Fold cross-validation       |
| `kfold_weighted_ensembled.py`    | Ensemble model using weighted average strategy   |
| `lightgbm.py`                    | LightGBM model experiment                        |
| `ngboost.py`                     | NGBoost regressor testing                        |
| `random_forest_regressor.py`     | Random Forest regressor experiment               |
| `test_score_xgboost.py`          | Evaluation of enhanced XGBoost model             |
| `test_score_lightgbm.py`         | Performance scoring of LightGBM model            |
| `test_score_ngboost.py`          | Performance evaluation of NGBoost                |
| `test_score_random_forest.py`    | Evaluation metrics for Random Forest             |
| `test_score_weighted_ensembled.py`| Final testing of weighted ensemble predictions   |
| `weighted_ensembled.py`          | Combines predictions from multiple models        |



Each script trains a model or evaluates it using scoring metrics like RMSE, MAE, R².

📌 These  are testing and experimentation scripts that guide model selection.
