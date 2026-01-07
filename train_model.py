import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FLIGHT PRICE PREDICTION - MLOps Pipeline")
print("=" * 70)

# Load data
print("\n[1/4] Loading and cleaning data...")
df = pd.read_csv('data/raw/Clean_Dataset.csv')
print(f"✓ Original shape: {df.shape}")

# Drop index column
df = df.drop('Unnamed: 0', axis=1)
df = df.dropna().drop_duplicates()
print(f"✓ After cleaning: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")

# Separate target
y = df['price']
X = df.drop('price', axis=1)

# Encode categorical
print("\n[2/4] Preprocessing...")
label_encoders = {}
for col in X.select_dtypes('object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scale numeric
scaler = StandardScaler()
numeric_cols = X.select_dtypes(['float64', 'int64']).columns.tolist()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"✓ Encoded {len(label_encoders)} categorical features")
print(f"✓ Scaled {len(numeric_cols)} numeric features")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"✓ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Train
print("\n[3/4] Training models...")
xgb = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
xgb_r2 = r2_score(y_val, y_pred)
print(f"✓ XGBoost - RMSE: ${xgb_rmse:.2f}, R²: {xgb_r2:.4f}")

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
rf_r2 = r2_score(y_val, y_pred_rf)
print(f"✓ Random Forest - RMSE: ${rf_rmse:.2f}, R²: {rf_r2:.4f}")

# Evaluate
print("\n[4/4] Test evaluation...")
y_test_pred = xgb.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"✓ Test RMSE: ${test_rmse:.2f}")
print(f"✓ Test MAE: ${test_mae:.2f}")
print(f"✓ Test R²: {test_r2:.4f}")

# Save with metadata
print("\n" + "=" * 70)
print("SAVING ARTIFACTS")
print("=" * 70)

pickle.dump(xgb, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(label_encoders, open('encoders.pkl', 'wb'))

# Save metadata
metadata = {
    'feature_names': X.columns.tolist(),
    'numeric_cols': numeric_cols,
    'categorical_cols': list(label_encoders.keys())
}
pickle.dump(metadata, open('metadata.pkl', 'wb'))

print("✓ model.pkl")
print("✓ scaler.pkl")
print("✓ encoders.pkl")
print("✓ metadata.pkl")
print("=" * 70)
print("✓ Complete!")
