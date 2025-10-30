import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Basic cleanup
df = df.dropna(subset=["location", "total_sqft", "bath", "size", "price"])
df = df[df["price"] > 0]

# Feature Engineering
df["BHK"] = df["size"].apply(lambda x: int(str(x).split(" ")[0]) if " " in str(x) else np.nan)
df["bath"] = df["bath"].astype(int)
df["balcony"] = df["balcony"].fillna(0).astype(int)
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")
df = df.dropna(subset=["total_sqft"])

# Select key features
features = ["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK"]
target = "price"

# Remove missing text
df = df.dropna(subset=["area_type", "society", "location"])

# -------------------------------------------------------
# Encoding categorical variables
# -------------------------------------------------------
categorical_features = ["area_type", "society", "location"]
numeric_features = ["total_sqft", "bath", "balcony", "BHK"]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

# -------------------------------------------------------
# Model definition
# -------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

accuracy = r2_score(y_test, preds)
print(f"✅ Model Training Completed with Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
joblib.dump(pipeline, "pipeline_model.pkl")
print("✅ Model saved as 'pipeline_model.pkl'")
