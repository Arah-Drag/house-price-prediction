# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Drop rows with missing values in key columns
df = df.dropna(subset=["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK", "price"])

# -------------------------------------------------------
# Features & Target
# -------------------------------------------------------
X = df[["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK"]]
y = df["price"]

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Preprocessing (OneHotEncoder for categorical)
# -------------------------------------------------------
categorical_features = ["area_type", "society", "location"]
numeric_features = ["total_sqft", "bath", "balcony", "BHK"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# -------------------------------------------------------
# Model: Random Forest
# -------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# -------------------------------------------------------
# Pipeline
# -------------------------------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# -------------------------------------------------------
# Train Model
# -------------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"✅ Model Trained Successfully | Accuracy (R²): {score * 100:.2f}%")

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
joblib.dump(pipeline, "pipeline_model.pkl")
print("💾 Model saved as pipeline_model.pkl")
