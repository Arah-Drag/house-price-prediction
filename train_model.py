# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Remove rows with any missing values
df = df.dropna(subset=["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK", "price"])

# -------------------------------------------------------
# Feature & Target Selection
# -------------------------------------------------------
X = df[["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK"]]
y = df["price"]

# -------------------------------------------------------
# Split Data
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Preprocessing: OneHotEncode categorical columns
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
model = RandomForestRegressor(n_estimators=200, random_state=42)

# -------------------------------------------------------
# Pipeline
# -------------------------------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# -------------------------------------------------------
# Train the Model
# -------------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# Evaluate the Model
# -------------------------------------------------------
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"✅ Model R² Accuracy: {score * 100:.2f}%")

# -------------------------------------------------------
# Save the Model
# -------------------------------------------------------
joblib.dump(pipeline, "pipeline_model.pkl")
print("💾 Model saved as pipeline_model.pkl")
