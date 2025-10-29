# train_model.py (improved for Bengaluru dataset)
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load
df = pd.read_csv("Bengaluru_House_Data.csv")

# 2. Basic cleaning and parsing functions
def parse_bhk(x):
    try:
        if isinstance(x, str):
            return int(x.split()[0])
        return np.nan
    except:
        return np.nan

def parse_total_sqft(x):
    try:
        if isinstance(x, str):
            x = x.strip()
            if '-' in x:
                a, b = x.split('-')
                return (float(a) + float(b)) / 2.0
            if x.replace('.', '', 1).isdigit():
                return float(x)
            # handle things like "34.46Sq. Meter" or "4120 sqft"
            nums = ''.join(ch if (ch.isdigit() or ch=='.' or ch=='-') else ' ' for ch in x).split()
            if len(nums) == 1:
                return float(nums[0])
            elif len(nums) >= 2:
                return np.mean([float(n) for n in nums if n])
        elif isinstance(x, (int, float)):
            return float(x)
    except:
        return np.nan
    return np.nan

# Apply parsing
df['bhk'] = df['size'].apply(parse_bhk)
df['total_sqft_num'] = df['total_sqft'].apply(parse_total_sqft)

# Drop rows with missing essential values
df = df.dropna(subset=['total_sqft_num', 'bhk', 'price'])

# Filter unrealistic per-BHK sqft
df = df[df['total_sqft_num'] / df['bhk'] >= 300]

# Keep top-K locations
K = 100
top_locations = df['location'].value_counts().nlargest(K).index.tolist()
df['location_clean'] = df['location'].apply(lambda x: x if x in top_locations else 'other')

# Feature selection
features = ['total_sqft_num', 'bath', 'balcony', 'bhk', 'area_type', 'location_clean']
X = df[features].copy()
y = np.log1p(df['price'])   # log transform target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
num_features = ['total_sqft_num', 'bath', 'balcony', 'bhk']
cat_features = ['area_type', 'location_clean']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Final pipeline with Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)    # back to original scale
y_test_orig = np.expm1(y_test)

rmse = mean_squared_error(y_test_orig, y_pred, squared=False)
r2 = r2_score(y_test_orig, y_pred)

print(f"RMSE (orig scale): {rmse:.2f}")
print(f"R2 (orig scale): {r2:.3f}")

# Save pipeline and top_locations
joblib.dump(pipeline, "pipeline_model.pkl")
joblib.dump(top_locations, "location_topk.pkl")
joblib.dump(features, "feature_names.pkl")

print("Saved: pipeline_model.pkl, location_topk.pkl, feature_names.pkl")
