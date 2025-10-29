import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------
# Load dataset
# --------------------------------------
data = pd.read_csv("Bengaluru_House_Data.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop duplicates and missing prices
data = data.drop_duplicates()
data = data.dropna(subset=['price'])

# Extract numeric and categorical columns
categorical_cols = ['location', 'area_type', 'availability']
numerical_cols = ['bath', 'balcony', 'size']

# Convert 'size' (e.g., "2 BHK") to numeric
def extract_bhk(x):
    try:
        return int(x.split()[0])
    except:
        return np.nan

data['size'] = data['size'].apply(extract_bhk)

# Drop rows with invalid values
data = data.dropna(subset=['size', 'bath', 'balcony'])

# Convert total_sqft to numeric (handles ranges like "1200-1500")
def convert_sqft(x):
    try:
        if '-' in str(x):
            low, high = map(float, x.split('-'))
            return (low + high) / 2
        else:
            return float(x)
    except:
        return np.nan

data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data = data.dropna(subset=['total_sqft'])

# Target variable
y = data['price']
X = data[['total_sqft', 'bath', 'balcony', 'size', 'location', 'area_type', 'availability']]

# --------------------------------------
# Preprocessing
# --------------------------------------
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['total_sqft', 'bath', 'balcony', 'size']),
        ('cat', categorical_transformer, ['location', 'area_type', 'availability'])
    ]
)

# --------------------------------------
# Build and Train Pipeline
# --------------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# --------------------------------------
# Evaluate Model
# --------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# --------------------------------------
# Save Model and Metadata
# --------------------------------------
joblib.dump(model, "pipeline_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("🎯 Model and feature names saved successfully.")
