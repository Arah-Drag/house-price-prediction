# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -------------------------------------------------------
# Load and clean dataset
# -------------------------------------------------------
df = pd.read_csv("Cleaned_Bengaluru_House_Data.csv")
df = df.dropna(subset=["size", "total_sqft", "bath", "price", "location"])

# Extract BHK
df["BHK"] = df["size"].apply(lambda x: int(x.split(" ")[0]) if isinstance(x, str) else 0)

# Convert total_sqft to numeric (handle ranges and text)
def convert_sqft(x):
    try:
        x = str(x).lower().replace("sq. meter", "").replace("acres", "").replace("cent", "").replace("perch", "")
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# Remove outliers based on price per sqft
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]
df = df[(df["price_per_sqft"] > 2000) & (df["price_per_sqft"] < 20000)]

# Simplify location
df["location"] = df["location"].apply(lambda x: x.strip())
location_stats = df["location"].value_counts()
df["location"] = df["location"].apply(lambda x: "other" if location_stats[x] <= 10 else x)

# Encode location
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

# -------------------------------------------------------
# Feature selection
# -------------------------------------------------------
X = df[["location", "total_sqft", "bath", "BHK"]]
y = df["price"]

# -------------------------------------------------------
# Split and train
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Model Trained Successfully!")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 MAE: {mae:.2f} lakhs")

# -------------------------------------------------------
# Save model
# -------------------------------------------------------
joblib.dump(model, "pipeline_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("🎯 Files saved successfully!")
