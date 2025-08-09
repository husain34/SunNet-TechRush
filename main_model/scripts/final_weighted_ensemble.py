import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from sklearn.model_selection import train_test_split

# Gradient boosting keeping both, haven’t decided which I like more yet
import xgboost as xgb
import lightgbm as lgb


#  Load data 
# Might need to wrap this in a try/except later
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Fill missing numeric values (quick and dirty)
train_df = train_df.fillna(train_df.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))

TARGET = 'power-generated'


#  Feature engineering 
def add_weather_features(df):
    # Temp squared — extreme heat has non-linear impact on panels
    df['temp_sq'] = df['temperature'] ** 2

    # Interaction term: wind and humidity together can mean fog or mist
    df['wind_x_humidity'] = df['wind-speed'] * df['humidity']

    # Approximation of sun angle (cosine transform)
    df['daylight_factor'] = np.maximum(
        0, np.cos(2 * np.pi * (df['distance-to-solar-noon'] - 0.5))
    )

    # Simple weather condition flags
    df['rain_or_fog'] = (
        (df['sky-cover'] >= 3) &
        (df['humidity'] >= 80) &
        (df['visibility'] <= 6)
    ).astype(int)

    # Guessing pollution presence
    df['pollution_guess'] = (
        (df['visibility'] < 7) &
        (df['average-pressure-(period)'] < 29.9)
    ).astype(int)

    # Too hot days  panels get lazy
    df['too_hot'] = (df['temperature'] > 30).astype(int)

    # Dew or frost risk early morning
    df['morning_dew_flag'] = (
        (df['temperature'] < 10) &
        (df['humidity'] > 90) &
        (df['distance-to-solar-noon'] < 0.2)
    ).astype(int)

    # Dropping without inplace  I always forget if this matters
    df.drop(columns=['distance-to-solar-noon'], inplace=False)
    return df


# Apply custom features
train_df = add_weather_features(train_df)
test_df = add_weather_features(test_df)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


#  Stage 1: Binary classification (power vs no power) 
y_train_binary = (y_train > 0).astype(int)

print("\n[Stage 1] Training classifier...")
clf_stage1 = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
)
clf_stage1.fit(X_train, y_train_binary)
binary_preds = clf_stage1.predict(X_test)


#  Stage 2: Regression models 
print("\n[Stage 2] Training regressors...")
rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
lgbm_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)

# Training  could be parallelized but meh
for model in (rf_model, xgb_model, lgbm_model):
    model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgbm_model.predict(X_test)

# Weighting  eyeballed, no hyperopt yet
w_rf, w_xgb, w_lgb = 0.2, 0.4, 0.4
ensemble_pred = (rf_pred * w_rf) + (xgb_pred * w_xgb) + (lgb_pred * w_lgb)

# Combine classifier + regressor
final_pred = np.where(binary_preds == 1, ensemble_pred, 0)
final_pred[final_pred < 0] = 0  # Avoid negative predictions


#  Evaluation 
r2 = r2_score(y_test, final_pred)
mae = mean_absolute_error(y_test, final_pred)
rmse = np.sqrt(mean_squared_error(y_test, final_pred))

print("\n📊 Model Performance:")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# Save everything for later 
model_bundle = {
    'classifier_stage1': clf_stage1,
    'rf': rf_model,
    'xgb': xgb_model,
    'lgb': lgbm_model,
    'weights': {'rf': w_rf, 'xgb': w_xgb, 'lgb': w_lgb},
    'feature_list': list(X_train.columns)
}

joblib.dump(model_bundle, 'two_stage_solar_model.pkl')
print("\n✅ Saved model as two_stage_solar_model.pkl")


#  Zero-prediction analysis 
n_zero_actual = np.sum(y_test == 0)
n_zero_pred = np.sum(final_pred == 0)
n_zero_correct = np.sum((y_test == 0) & (final_pred == 0))
false_pos = np.sum((y_test == 0) & (final_pred > 0))
false_neg = np.sum((y_test > 0) & (final_pred == 0))

print("\n🔍 Zero Prediction Analysis:")
print(f"Actual zeros       : {n_zero_actual}")
print(f"Predicted zeros    : {n_zero_pred}")
print(f"Correct zero preds : {n_zero_correct}")
print(f"False positives    : {false_pos}")
print(f"False negatives    : {false_neg}")


#  Plots 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_pred, alpha=0.4, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect")
plt.xlabel("Actual Power")
plt.ylabel("Predicted Power")
plt.title("Predicted vs Actual (Two-Stage Model)")
plt.legend()
plt.grid(True)
plt.show()

# Residuals
residuals = y_test - final_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_test, residuals, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Power")
plt.ylabel("Residual")
plt.title("Residuals vs Actual Power")
plt.grid(True)
plt.show()

# Error distribution
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Residual")
plt.grid(True)
plt.show()
