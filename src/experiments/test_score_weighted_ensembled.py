import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load the ensemble model and data ===
model = joblib.load("weighted_ensemble_model.pkl")
data = pd.read_csv("solarpowergeneration.csv").dropna()

X = data[model['columns']].values
y_true = data['power-generated'].values

# === Predict using the saved ensemble model ===
pred_rf = model['rf'].predict(X)
pred_xg = model['xg'].predict(X)
w_rf = model['weights']['rf']
w_xg = model['weights']['xg']
y_pred = (w_rf * pred_rf) + (w_xg * pred_xg)

# === Clip negative predictions and actuals ===
y_pred = np.clip(y_pred, 0, None)
y_true = np.clip(y_true, 0, None)

# === Plot Actual vs Predicted ===
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions', color='cornflowerblue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.title('Predicted vs. Actual Power Generated')
plt.xlabel('Actual Power Generated')
plt.ylabel('Predicted Power Generated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Zero-actual value error analysis ===
zero_actual_indices = (y_true == 0)
num_zeros = np.sum(zero_actual_indices)

if num_zeros > 0:
    pred_when_zero = y_pred[zero_actual_indices]
    avg_error_when_zero = np.mean(np.abs(pred_when_zero))

    print(f"\n🔍 When actual = 0 ({num_zeros} samples):")
    print(f"    - Avg Prediction: {np.mean(pred_when_zero):.4f}")
    print(f"    - Max Prediction: {np.max(pred_when_zero):.4f}")
    print(f"    - Min Prediction: {np.min(pred_when_zero):.4f}")
    print(f"    - Avg Absolute Error: {avg_error_when_zero:.4f}")

    # === Plot predictions for zero-actual samples ===
    plt.figure(figsize=(10, 5))
    plt.plot(pred_when_zero, 'o-', color='orange', label='Predicted Value')
    plt.axhline(0, color='red', linestyle='--', label='True Value (0)')
    plt.title('Predictions When Actual Power = 0')
    plt.xlabel('Sample Index (only where actual = 0)')
    plt.ylabel('Predicted Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("\n✅ No samples where actual value = 0.")

# === Standard Regression Metrics ===
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📏 Standard Regression Metrics:")
print(f"    - MAE  : {mae:.4f}")
print(f"    - RMSE : {rmse:.4f}")
print(f"    - R²   : {r2:.4f}")

# === SMAPE ===
epsilon = 1e-5
smape = 100 * np.mean(
    2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon)
)
print(f"\n🔁 SMAPE: {smape:.2f}%")

# === Logarithmic Error ===
log_error = np.mean(np.abs(np.log1p(y_pred + epsilon) - np.log1p(y_true + epsilon)))
print(f"🧮 Mean Logarithmic Error: {log_error:.4f}")

# === Custom Scale-Sensitive Weighted Error ===
scale_sensitive_weight = 1 / (np.abs(y_true) + 1)
scale_sensitive_weight = np.clip(scale_sensitive_weight, 0, 1)  # Optional cap
scaled_error = np.abs(y_true - y_pred) * scale_sensitive_weight
weighted_avg_scaled_error = np.mean(scaled_error)
print(f"📐 Weighted Scaled Error: {weighted_avg_scaled_error:.2f}")
