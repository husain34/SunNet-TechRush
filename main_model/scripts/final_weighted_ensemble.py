import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

train_data = train_data.fillna(train_data.mean(numeric_only=True))
test_data = test_data.fillna(test_data.mean(numeric_only=True))
label = 'power-generated'

def feature_engineering(df):

    # Weather Relations with data from dataset for accurate
    df["temp_squared"] = df["temperature"] ** 2  # Extreme heat effects
    df["wind_speed_humidity"] = df["wind-speed"] * df["humidity"]  # Mist

    df["daylight_factor"] = np.maximum(0, np.cos(2 * np.pi * (df["distance-to-solar-noon"] - 0.5)))  # Approximates Sun Angle and peak at noon

    df["rain_or_fog_likelihood"] = ((df["sky-cover"] >= 3) & (df["humidity"] >= 80) & (df["visibility"] <= 6)).astype(int)  # Flags potential rain or fog conditions

    df["pollution_proxy"] = ((df["visibility"] < 7) & (df["average-pressure-(period)"] < 29.9)).astype(int)  # Poor visibility and low pressure — likely pollution or haze

    df["overheat_flag"] = (df["temperature"] > 30).astype(int)  # Very hot days — potential panel efficiency drop

    df["dew_morning_risk"] = ((df["temperature"] < 10) & (df["humidity"] > 90) & (df["distance-to-solar-noon"] < 0.2) ).astype(int)  # Cold, humid early mornings — possible dew or frost risk

    df.drop(columns=["distance-to-solar-noon"])

    return df

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

X_train = train_data.drop(columns=[label])
X_test = test_data.drop(columns=[label])
y_train = train_data[label]
y_test = test_data[label]

# Classification Model
y_train_binary = (y_train > 0).astype(int)

print("\n Binary Classifier (Power Generated vs. No Power)")
classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
classifier.fit(X_train, y_train_binary)
binary_predictions = classifier.predict(X_test)

# --- Stage 2: Regression Model
print("\n Ensemble Regressor ")
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
xg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)
xg.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_xg = xg.predict(X_test)
pred_lgbm = lgbm.predict(X_test)
w_rf = 0.2
w_xg = 0.4
w_lgb = 0.4
ensemble_reg_preds = (w_rf * pred_rf) + (w_xg * pred_xg) + (w_lgb * pred_lgbm)

final_ensemble_preds = np.where(binary_predictions == 1, ensemble_reg_preds, 0)
final_ensemble_preds[final_ensemble_preds < 0] = 0


r2 = r2_score(y_test, final_ensemble_preds)
mae = mean_absolute_error(y_test, final_ensemble_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_ensemble_preds))

print("\n Scores \n")
print(f"Two-Stage Ensemble R² Score  : {r2:.4f}")
print(f"Two-Stage Ensemble MAE       : {mae:.2f}")
print(f"Two-Stage Ensemble RMSE      : {rmse:.2f}")

ensemble_model = {
    'classifier': classifier,
    'rf': rf,
    'xg': xg,
    'lgb': lgbm,
    'weights': {'rf': w_rf, 'xg': w_xg, 'lgb': w_lgb},
    'columns': list(X_train.columns)
}

# Save to pkl
joblib.dump(ensemble_model, 'weighted_ensemble_model.pkl')
print("\n✅ Saved integrated two-stage model as two_stage_power_model_integrated.pkl")

# Load Data
model_data = joblib.load("weighted_ensemble_model.pkl")

classifier = model_data['classifier']
rf = model_data['rf']
xg = model_data['xg']
lgb = model_data.get('lgb')
weights = model_data['weights']
feature_columns = model_data['columns']

test_data = pd.read_csv("test_data.csv")
test_data = test_data.fillna(test_data.mean(numeric_only=True))

def feature_engineering(df):

    # Weather Relations with data from dataset for accurate
    df["temp_squared"] = df["temperature"] ** 2  # Extreme heat effects
    df["wind_speed_humidity"] = df["wind-speed"] * df["humidity"]  # Mist

    df["daylight_factor"] = np.maximum(0, np.cos(2 * np.pi * (df["distance-to-solar-noon"] - 0.5)))  # Approximates Sun Angle and peak at noon

    df["rain_or_fog_likelihood"] = ((df["sky-cover"] >= 3) & (df["humidity"] >= 80) & (df["visibility"] <= 6)).astype(int)  # Flags potential rain or fog conditions

    df["pollution_proxy"] = ((df["visibility"] < 7) & (df["average-pressure-(period)"] < 29.9)).astype(int)  # Poor visibility and low pressure likely pollution or haze

    df["overheat_flag"] = (df["temperature"] > 30).astype(int)  # Very hot days potential panel efficiency drop

    df["dew_morning_risk"] = ((df["temperature"] < 10) & (df["humidity"] > 90) & (df["distance-to-solar-noon"] < 0.2)).astype(int)  # Cold, humid early mornings possible dew or frost risk

    df.drop(columns=["distance-to-solar-noon"])
    return df


test_data = feature_engineering(test_data)
X_test = test_data[feature_columns]
y_test = test_data['power-generated']

# Binary Prediction
binary_predictions = classifier.predict(X_test)

pred_rf = rf.predict(X_test)
pred_xg = xg.predict(X_test)
pred_lgb = lgb.predict(X_test) if lgb else np.zeros_like(pred_rf)

ensemble_reg_preds = (
    weights['rf'] * pred_rf +
    weights['xg'] * pred_xg +
    weights.get('lgb', 0.0) * pred_lgb
)

final_ensemble_preds = np.where(binary_predictions == 1, ensemble_reg_preds, 0)
final_ensemble_preds[final_ensemble_preds < 0] = 0

#Metrics
mae = mean_absolute_error(y_test, final_ensemble_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_ensemble_preds))
r2 = r2_score(y_test, final_ensemble_preds)

print(f"\n📊 Model Evaluation (Two-Stage Ensemble):")
print(f"R² Score  : {r2:.4f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")

#Analysis of Zero
num_true_zeros = np.sum(y_test == 0)
num_predicted_zeros = np.sum(final_ensemble_preds == 0)
num_correctly_predicted_zeros = np.sum((y_test == 0) & (final_ensemble_preds == 0))
num_false_positives = np.sum((y_test == 0) & (final_ensemble_preds > 0))
num_false_negatives = np.sum((y_test > 0) & (final_ensemble_preds == 0))

print(f"\nAnalysis of Zero Predictions")
print(f"Total actual zero power instances in test set: {num_true_zeros}")
print(f"Total predicted zero power instances: {num_predicted_zeros}")
print(f"Correctly predicted zero power instances: {num_correctly_predicted_zeros}")
print(f"False Positives (predicted power, but actual was zero): {num_false_positives}")
print(f"False Negatives (predicted zero, but actual was power): {num_false_negatives}")


# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_ensemble_preds, alpha=0.4, label="Predictions") # Use final_ensemble_preds
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Power Generated")
plt.ylabel("Predicted Power Generated")
plt.title("Predicted vs. Actual Power Generated (Two-Stage Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - final_ensemble_preds # Use final_ensemble_preds
plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Power Generated")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Actual Power Generated (Two-Stage Model)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Error distribution ===
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Distribution of Prediction Errors (Two-Stage Model)")
plt.xlabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()