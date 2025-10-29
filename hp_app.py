# hp_app.py (updated to load pipeline_model.pkl)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# Load pipeline and location list
try:
    pipeline = joblib.load("pipeline_model.pkl")
    top_locations = joblib.load("location_topk.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"Required files not found: {e}")
    st.stop()

st.set_page_config(page_title="🏠 Bengaluru House Price DSS", layout="wide")
st.title("🏠 Bengaluru House Price Prediction DSS")

# Sidebar inputs
st.sidebar.header("Enter property details")
# Use top_locations for dropdown
loc = st.sidebar.selectbox("Location (area)", top_locations + ['other'])
area_type = st.sidebar.selectbox("Area Type", ['Super built-up  Area','Built-up  Area','Carpet  Area','Plot  Area'])
size = st.sidebar.text_input("Size (e.g. 2 BHK)", "2 BHK")
total_sqft = st.sidebar.text_input("Total sqft (e.g. 1050 or 1050 - 1200)", "1050")
bath = st.sidebar.number_input("Bath", min_value=0, max_value=10, value=2)
balcony = st.sidebar.number_input("Balcony", min_value=0, max_value=5, value=1)

# parse size and total_sqft here (use same functions as train)
def parse_bhk(x):
    try:
        if isinstance(x, str):
            return int(x.split()[0])
    except:
        return np.nan

def parse_total_sqft(x):
    try:
        if isinstance(x, str):
            x = x.strip()
            if '-' in x:
                a,b = x.split('-')
                return (float(a)+float(b))/2.0
            if x.replace('.', '', 1).isdigit():
                return float(x)
            nums = ''.join(ch if (ch.isdigit() or ch=='.' or ch=='-') else ' ' for ch in x).split()
            if len(nums)>=1:
                return float(nums[0])
        elif isinstance(x,(int,float)):
            return float(x)
    except:
        return np.nan
    return np.nan

bhk = parse_bhk(size)
total_sqft_num = parse_total_sqft(total_sqft)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'total_sqft_num': total_sqft_num,
    'bath': bath,
    'balcony': balcony,
    'bhk': bhk,
    'area_type': area_type,
    'location_clean': loc
}])

# Predict button
if st.button("Predict Price"):
    # Basic validation
    if pd.isna(total_sqft_num) or pd.isna(bhk):
        st.error("Please enter valid 'size' and 'total_sqft' values (e.g., '2 BHK' and '1050').")
    else:
        # clip unrealistic values
        input_df['total_sqft_num'] = np.clip(input_df['total_sqft_num'], 300, 10000)
        input_df['bhk'] = np.clip(input_df['bhk'], 1, 20)

        # pipeline predicts log price, so convert back
        pred_log = pipeline.predict(input_df)[0]
        pred_price = np.expm1(pred_log)
        pred_price = max(pred_price, 10000)  # safety floor

        st.success(f"Estimated Price: ₹ {pred_price:,.2f}")
        st.info("Note: price shown in INR (original scale). This model was trained on Bengaluru dataset.")

# Chatbot (brief)
st.write("---")
st.subheader("Chatbot")
q = st.text_input("Ask about dataset or features:")
if q:
    q = q.lower()
    if 'bhk' in q or 'size' in q:
        st.write("BHK is extracted from size (e.g., '2 BHK' → 2).")
    elif 'sqft' in q or 'total' in q:
        st.write("Total sqft is parsed; ranges like '1000 - 1200' are averaged.")
    else:
        st.write("Try asking about 'bhk', 'sqft', 'location'.")
