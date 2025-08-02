import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("lightgbm_model.pkl")

# Load the test data
data = pd.read_csv("solarpowergeneration.csv")
data = data.dropna()  # Ensure no NaNs

# Set target and features properly
target = 'power-generated'
features = [col for col in data.columns if col != target]

# Prepare inputs
X = data[features].values
y_true = data[target].values

# Predict using the saved model
y_pred = model.predict(X)

# === Error percentage ===
epsilon = 1e-5
percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
weights = y_true
weighted_avg_error = np.average(percentage_error, weights=weights) * 100
print(f"📊 Weighted Average Percentage Error: {weighted_avg_error:.2f}%")

# === Error when actual output is zero ===
zero_actual_indices = (y_true == 0)
num_zeros = np.sum(zero_actual_indices)

pred_when_zero = y_pred[zero_actual_indices]
avg_error_when_zero = np.mean(np.abs(pred_when_zero))

print(f"\n🔍 When actual = 0 ({num_zeros} samples):")
print(f"    - Avg Prediction: {np.mean(pred_when_zero):.4f}")
print(f"    - Max Prediction: {np.max(pred_when_zero):.4f}")
print(f"    - Min Prediction: {np.min(pred_when_zero):.4f}")
print(f"    - Avg Absolute Error: {avg_error_when_zero:.4f}")

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