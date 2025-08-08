import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import joblib

# Load the dataset
data = pd.read_csv("solarpowergeneration.csv").dropna()

data["sin_time"] = np.sin(2 * np.pi * data["distance-to-solar-noon"])
data["cos_time"] = np.cos(2 * np.pi * data["distance-to-solar-noon"])
data["temp_squared"] = data["temperature"] ** 2
data["wind_speed_humidity"] = data["wind-speed"] * data["humidity"]
data["pressure_humidity"] = data["average-pressure-(period)"] * data["humidity"]

# Split features and target
X = data.drop(columns=['power-generated']).values
y = data['power-generated'].values

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

print("🔁 Performing 5-Fold Cross-Validation...\n")
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f"Fold {fold} R² Score: {r2:.4f}")

# Average R² Score across folds
avg_r2 = np.mean(r2_scores)
print(f"\n📊 Average R² Score across 5 folds: {avg_r2:.4f}")

# Train final model on full dataset
final_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model, 'xgboost_model.pkl')
print("✅ Final model trained and saved as 'xgboost_model.pkl'")