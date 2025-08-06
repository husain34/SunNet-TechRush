import pandas as pd
import numpy as np
import ngboost as ngb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import joblib

# Load the dataset
data = pd.read_csv("solarpowergeneration.csv").dropna()

# Split features and target
X = data.drop(columns=['power-generated']).values
y = data['power-generated'].values

# KFold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

# Cross-validation
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = ngb.NGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f"Fold R² Score: {r2:.3f}")

# Final R² score
print(f"\nAverage R² Score across all folds: {np.mean(r2_scores):.3f}")

# Train final model on full dataset
final_model = ngb.NGBRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model,'lightgbm_model.pkl')

print("✅ Model saved as xgboost_model.pkl")
