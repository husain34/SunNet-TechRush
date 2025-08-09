# SunNet-TechRush
A machine learning model to predict solar energy generation using historical weather data. Incorporates features like temperature, cloud cover, and seasonal patterns for accurate forecasting. Built by Team SunNet for TechRush Hackathon, PISB focused on sustainable energy solutions.

## 📌 Features
- Multiple models : Includes various regression algorithms for experimentation.
- Data Handling : Works with raw training and testing datasets.
- Visualization : Generates charts, graphs, and performance evaluation plots.
  
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


## 📁 Project Structure

- `data/`
  - `raw/`: Contains the original input datasets used for model training and evaluation.
    - `train.csv`: Dataset used for training models.
    - `test.csv`: Dataset used for testing/inference.
    - `solarpowergeneration.csv`: Full dataset with solar energy generation and weather features.

- `models/`:  
  Contains saved machine learning models in `.pkl` format, such as XGBoost, and ensemble model.

- `src/`:
  - `models/`: Python scripts for training and saving models.
  - `experiments/`: Scripts for running different model experiments, performance evaluation, and visualization.

- `images/`:  
  Stores graphs, charts, and visualizations such as prediction vs actual plots, error analysis, and evaluation results.
  
- `ui/`:  
  Contains early user interface prototypes for the project.  
  - `prototype_ui.py`: A layout-only Streamlit app used to visualize the intended structure of the user interface (does not yet connect to   trained models).


- `README.md`:  
  Project overview, structure, and updates.

