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

# Weighted ensemble 
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


"""WEIGHTED ENSEMBLE (KFOLD)"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv("solarpowergeneration.csv")
data = data.dropna()

# Separate input features and target
X = data.drop(columns=['power-generated']).values
y = data['power-generated'].values

# Set the weights for the ensemble
w_rf = 0.1
w_xg = 0.9

# Create the KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store R2 scores for each fold
r2_scores = []

# Loop through each fold
for train_index, test_index in kf.split(X):
    # Split into training and testing based on the fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create and train models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xg = xgb.XGBRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    xg.fit(X_train, y_train)

    # Make predictions
    pred_rf = rf.predict(X_test)
    pred_xg = xg.predict(X_test)

    # Combine predictions using weights
    ensemble_preds = (w_rf * pred_rf) + (w_xg * pred_xg)

    # Calculate R² score and store
    r2 = r2_score(y_test, ensemble_preds)
    r2_scores.append(r2)

    print(f"Fold R² Score: {r2:.3f}")

# Print average R² score across all folds
print(f"\nAverage R² Score across all folds: {np.mean(r2_scores):.3f}")

# Train final models on the full dataset
rf.fit(X, y)
xg.fit(X, y)

# Save the ensemble
ensemble_model = {
    'rf': rf,
    'xg': xg,
    'weights': {'rf': w_rf, 'xg': w_xg},
    'columns': list(data.drop(columns=['power-generated']).columns)
}

joblib.dump(ensemble_model, 'weighted_ensemble_model.pkl')
print("✅ Model saved as weighted_ensemble_model.pkl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the ensemble model and data
model = joblib.load("weighted_ensemble_model.pkl")
data = pd.read_csv("solarpowergeneration.csv").dropna()

X = data[model['columns']].values
y_true = data['power-generated'].values

#Predict using the saved ensemble model
pred_rf = model['rf'].predict(X)
pred_xg = model['xg'].predict(X)
w_rf = model['weights']['rf']
w_xg = model['weights']['xg']
y_pred = (w_rf * pred_rf) + (w_xg * pred_xg)

# Clip negative predictions and actuals
y_pred = np.clip(y_pred, 0, None)
y_true = np.clip(y_true, 0, None)

#Plot Actual vs Predicted
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

# Zero-actual value error analysis
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

    # Plot predictions for zero-actual samples
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

#  Standard Regression Metrics 
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📏 Standard Regression Metrics:")
print(f"    - MAE  : {mae:.4f}")
print(f"    - RMSE : {rmse:.4f}")
print(f"    - R²   : {r2:.4f}")

# SMAPE
epsilon = 1e-5
smape = 100 * np.mean(
    2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon)
)
print(f"\n🔁 SMAPE: {smape:.2f}%")

#  Logarithmic Error
log_error = np.mean(np.abs(np.log1p(y_pred + epsilon) - np.log1p(y_true + epsilon)))
print(f"🧮 Mean Logarithmic Error: {log_error:.4f}")

#  Custom Scale-Sensitive Weighted Error
scale_sensitive_weight = 1 / (np.abs(y_true) + 1)
scale_sensitive_weight = np.clip(scale_sensitive_weight, 0, 1)  # Optional cap
scaled_error = np.abs(y_true - y_pred) * scale_sensitive_weight
weighted_avg_scaled_error = np.mean(scaled_error)
print(f"📐 Weighted Scaled Error: {weighted_avg_scaled_error:.2f}")

"""XGBOOST KFOLD MODE"""

import pandas as pd
import numpy as np
import xgboost as xgb
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

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f"Fold R² Score: {r2:.3f}")

# Final R² score
print(f"\nAverage R² Score across all folds: {np.mean(r2_scores):.3f}")

# Train final model on full dataset
final_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model,'xgboost_model.pkl')

print("✅ Model saved as xgboost_model.pkl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("xgboost_model.pkl")

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

# Ensure y_pred and y_true have the same shape
if len(y_true) != len(y_pred):
    print(f"❌ Length mismatch! y_true: {len(y_true)}, y_pred: {len(y_pred)}")
    exit()

# Error percentage 
epsilon = 1e-5
percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
weights = y_true
weighted_avg_error = np.average(percentage_error, weights=weights) * 100
print(f"📊 Weighted Average Percentage Error: {weighted_avg_error:.2f}%")

# Error when actual output is zero 
zero_actual_indices = (y_true == 0)
num_zeros = np.sum(zero_actual_indices)

pred_when_zero = y_pred[zero_actual_indices]
avg_error_when_zero = np.mean(np.abs(pred_when_zero))

print(f"\n🔍 When actual = 0 ({num_zeros} samples):")
print(f"    - Avg Prediction: {np.mean(pred_when_zero):.4f}")
print(f"    - Max Prediction: {np.max(pred_when_zero):.4f}")
print(f"    - Min Prediction: {np.min(pred_when_zero):.4f}")
print(f"    - Avg Absolute Error: {avg_error_when_zero:.4f}")

# Plot Actual vs Predicted 
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



    # Plot predictions for zero-actual samples 
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

"""RANDOM FOREST"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Step 1: Load and clean the data
df = pd.read_csv('solarpowergeneration.csv')

# Step 2: Select features and target
target = 'power-generated'
features = [col for col in df.columns if col != target]

X = df[features].values
y = df[target].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get individual predictions
pred_rf = rf.predict(X_test)

r2 = r2_score(y_test, pred_rf)

print(f"\nAverage R² Score across all folds: {r2:.3f}")

#Train final model on full data and save
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)
joblib.dump(final_model, 'random_forest_model.pkl')
print("✅ Final model trained on full data and saved as 'random_forest_model.pkl'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

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

#Error percentage 
epsilon = 1e-5
percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
weights = y_true
weighted_avg_error = np.average(percentage_error, weights=weights) * 100
print(f"📊 Weighted Average Percentage Error: {weighted_avg_error:.2f}%")

# Plot Actual vs Predicted
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

# Error when actual output is zero
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

    #Plot predictions for zero-actual samples 
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

"""LightGBM MODEL"""

import pandas as pd
import numpy as np
import lightgbm as lb
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

    model = lb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f"Fold R² Score: {r2:.3f}")

# Final R² score
print(f"\nAverage R² Score across all folds: {np.mean(r2_scores):.3f}")

# Train final model on full dataset
final_model = lb.LGBMRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Save the model
joblib.dump(final_model,'lightgbm_model.pkl')

print("✅ Model saved as xgboost_model.pkl")

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

#Error when actual output is zero
zero_actual_indices = (y_true == 0)
num_zeros = np.sum(zero_actual_indices)

pred_when_zero = y_pred[zero_actual_indices]
avg_error_when_zero = np.mean(np.abs(pred_when_zero))

print(f"\n🔍 When actual = 0 ({num_zeros} samples):")
print(f"    - Avg Prediction: {np.mean(pred_when_zero):.4f}")
print(f"    - Max Prediction: {np.max(pred_when_zero):.4f}")
print(f"    - Min Prediction: {np.min(pred_when_zero):.4f}")
print(f"    - Avg Absolute Error: {avg_error_when_zero:.4f}")

#Plot Actual vs Predicted
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



    #Plot predictions for zero-actual samples
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

#Error percentage
epsilon = 1e-5
percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))
weights = y_true
weighted_avg_error = np.average(percentage_error, weights=weights) * 100
print(f"📊 Weighted Average Percentage Error: {weighted_avg_error:.2f}%")

#Error when actual output is zero
zero_actual_indices = (y_true == 0)
num_zeros = np.sum(zero_actual_indices)

pred_when_zero = y_pred[zero_actual_indices]
avg_error_when_zero = np.mean(np.abs(pred_when_zero))

print(f"\n🔍 When actual = 0 ({num_zeros} samples):")
print(f"    - Avg Prediction: {np.mean(pred_when_zero):.4f}")
print(f"    - Max Prediction: {np.max(pred_when_zero):.4f}")
print(f"    - Min Prediction: {np.min(pred_when_zero):.4f}")
print(f"    - Avg Absolute Error: {avg_error_when_zero:.4f}")

#Plot Actual vs Predicted
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



    #Plot predictions for zero-actual samples
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
