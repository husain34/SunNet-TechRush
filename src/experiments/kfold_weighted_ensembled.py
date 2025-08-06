#WEIGHTED ENSEMBLE (NO KFOLD)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("solarpowergeneration.csv")
data = data.fillna(data.mean(numeric_only=True))

X = data.drop(columns=['power-generated'])
y = data['power-generated']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg.fit(X_train, y_train)

# Get individual predictions
pred_rf = rf.predict(X_test)
pred_xg = xg.predict(X_test)

# Weighted ensemble (you can adjust the weights)
w_rf = 0.1
w_xg = 0.9
ensemble_preds = (w_rf * pred_rf) + (w_xg * pred_xg)

# Evaluate ensemble
r2 = r2_score(y_test, ensemble_preds)
print(f"Weighted Ensemble R² Score: {r2:.3f}")

# Save all models and weights into one object
ensemble_model = {
    'rf': rf,
    'xg': xg,
    'weights': {'rf': w_rf, 'xg': w_xg},
    'columns': list(X.columns)
}

# Save to pkl
joblib.dump(ensemble_model, 'weighted_ensemble_model.pkl')
print("✅ Saved as weighted_ensemble_model.pkl")
