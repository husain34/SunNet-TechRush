# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

df = pd.read_csv('solarpowergeneration.csv')

target = 'power-generated'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='RMSE',
    eval_metric='R2',
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("mae = " , mae)
print("rmse = ", rmse)

plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("CatBoost: Predicted vs Actual Solar Power Generation")
plt.grid(True)
plt.show()

residuals = y_test.values - y_pred

plt.figure(figsize=(10, 3))
plt.plot(residuals, color='darkorange')
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals (Actual - Predicted)")
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.grid(True)
plt.tight_layout()
plt.show()

sns.histplot(residuals, kde=True, color='teal')
plt.title("Distribution of Residuals")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted (Scatter)")
plt.xlabel("Actual Power")
plt.ylabel("Predicted Power")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get raw feature importances from CatBoost
raw_importances = model.get_feature_importance()
normalized_importances = raw_importances / np.sum(raw_importances)

# Get feature names
feature_names = X.columns

# Plot normalized feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, normalized_importances, color='salmon')
plt.xlabel("Normalized Importance (0–1 scale)")
plt.title("Normalized Feature Importance (CatBoost)")
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

r2 = r2_score(y_test, y_pred)

print("🔍 Model Performance Summary (CatBoost)")
print(f"📈 MAE  (Mean Absolute Error): {mae:.2f}")
print(f"📉 RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"🎯 R² Score: {r2:.4f}")
print(f"✅ Accuracy-like Score: {r2 * 100:.2f}%")

