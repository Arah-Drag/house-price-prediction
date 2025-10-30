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

# -------------------------------------------------------
# Data Cleaning & Feature Engineering
# -------------------------------------------------------

# Drop rows with missing essential fields
df = df.dropna(subset=["area_type", "society", "location", "total_sqft", "bath", "balcony", "size", "price"])

# Extract BHK number from 'size' column (e.g., "2 BHK" -> 2)
df["BHK"] = df["size"].apply(lambda x: int(str(x).split()[0]) if isinstance(x, str) else None)

# Remove rows with invalid or 0 BHK
df = df[df["BHK"] > 0]

# Remove rows with non-numeric or invalid total_sqft
def clean_sqft(x):
    try:
        if isinstance(x, str) and "-" in x:
            nums = [float(i) for i in x.split("-")]
            return sum(nums) / len(nums)
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(clean_sqft)
df = df.dropna(subset=["total_sqft"])

# Remove extreme outliers
df = df[(df["price"] < 500) & (df["total_sqft"] < 10000)]

# -------------------------------------------------------
# Select Features & Target
# -------------------------------------------------------
X = df[["area_type", "society", "location", "total_sqft", "bath", "balcony", "BHK"]]
y = df["price"]

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Preprocessing Pipeline
# -------------------------------------------------------
categorical_features = ["area_type", "society", "location"]
numerical_features = ["total_sqft", "bath", "balcony", "BHK"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])

# -------------------------------------------------------
# Random Forest Model
# -------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

# -------------------------------------------------------
# Build Full Pipeline
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
# Evaluate Model
# -------------------------------------------------------
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"✅ Model Trained Successfully | Accuracy (R²): {score * 100:.2f}%")

# -------------------------------------------------------
# Save Model
# -------------------------------------------------------
joblib.dump(pipeline, "pipeline_model.pkl")
print("💾 Model saved as pipeline_model.pkl")
