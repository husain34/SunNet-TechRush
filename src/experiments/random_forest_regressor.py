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

# Step 7: Train final model on full data and save
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)
joblib.dump(final_model, 'random_forest_model.pkl')
print("✅ Final model trained on full data and saved as 'random_forest_model.pkl'")

