# ==========================================
#  train_model.py
#  House Price Prediction Model Training
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_excel("housedata.xlsx")

print("✅ Dataset loaded successfully!")
print("Columns:", data.columns.tolist())
print("Shape:", data.shape)

# -----------------------------
# 2. Handle missing values
# -----------------------------
data = data.dropna(subset=["Price"])  # ensure target column has no missing
data.fillna({
    "Car": data["Car"].median(),
    "BuildingArea": data["BuildingArea"].median(),
    "YearBuilt": data["YearBuilt"].median()
}, inplace=True)

# Drop non-useful columns
data = data.drop(["Address", "Date"], axis=1, errors='ignore')  # ignore if columns not found

# -----------------------------
# 3. Encode categorical columns
# -----------------------------
cat_cols = data.select_dtypes(include=["object"]).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# -----------------------------
# 4. Split features and target
# -----------------------------
X = data.drop("Price", axis=1)
y = data["Price"]

# Save feature names for later use
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# -----------------------------
# 7. Save model, encoders, and metadata
# -----------------------------
joblib.dump(model, "house_price_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")  # Save column names for prediction

print("💾 Saved:")
print("   ✔️ house_price_model.pkl")
print("   ✔️ encoders.pkl")
print("   ✔️ feature_names.pkl")
