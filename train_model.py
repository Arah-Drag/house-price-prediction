# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -------------------------------------------------------
# 1️⃣ Load and Clean Data
# -------------------------------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Drop rows with missing key values
df = df.dropna(subset=["size", "total_sqft", "bath", "price", "location"])

# Extract BHK from size
df["BHK"] = df["size"].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0)

# Convert total_sqft — handle ranges like "2100-2850"
def convert_sqft(x):
    try:
        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        else:
            return float(x)
    except:
        return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# Clean location values
df["location"] = df["location"].apply(lambda x: x.strip())
location_stats = df["location"].value_counts()
location_stats_less_than_10 = location_stats[location_stats <= 10]
df["location"] = df["location"].apply(lambda x: "other" if x in location_stats_less_than_10 else x)

# Encode locations
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

# -------------------------------------------------------
# 2️⃣ Feature Selection
# -------------------------------------------------------
X = df[["location", "total_sqft", "bath", "BHK"]]
y = df["price"]

# -------------------------------------------------------
# 3️⃣ Split and Train RandomForest
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)

rf.fit(X_train, y_train)

# -------------------------------------------------------
# 4️⃣ Evaluate Model
# -------------------------------------------------------
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Random Forest Model Trained Successfully!")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 Mean Absolute Error: {mae:.3f} Lakhs")

# -------------------------------------------------------
# 5️⃣ Save Model and Encoder
# -------------------------------------------------------
joblib.dump(rf, "pipeline_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n🎯 Files saved: pipeline_model.pkl and label_encoder.pkl")
