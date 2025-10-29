import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ------------------------------------------
# Load model and encoders
# ------------------------------------------
try:
    model = joblib.load("house_price_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"❌ Required files not found: {e}")
    st.stop()

# ------------------------------------------
# Page Configuration
# ------------------------------------------
st.set_page_config(page_title="🏠 House Price DSS", layout="wide")
st.title("🏠 AI-powered House Price Prediction DSS")
st.write("An intelligent Decision Support System (DSS) for property price forecasting using Machine Learning and AI Chatbot support.")

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("🔧 Enter Property Details")

inputs = {}

# Example dropdowns — replace with your dataset’s unique values for realism
suburbs = ["Richmond", "Abbotsford", "Carlton", "Fitzroy", "Melbourne"]
types = ["h", "u", "t"]
methods = ["S", "SP", "VB", "PI"]
sellergs = ["Biggin", "Nelson", "Jellis", "Hocking"]
council_areas = ["Yarra", "Melbourne City", "Moreland", "Darebin"]
regions = ["Northern Metropolitan", "Southern Metropolitan", "Eastern Metropolitan"]

inputs['Suburb'] = st.sidebar.selectbox("Suburb", suburbs)
inputs['Type'] = st.sidebar.selectbox("Type (h/u/t)", types)
inputs['Method'] = st.sidebar.selectbox("Sale Method", methods)
inputs['SellerG'] = st.sidebar.selectbox("Seller Group", sellergs)
inputs['CouncilArea'] = st.sidebar.selectbox("Council Area", council_areas)
inputs['Regionname'] = st.sidebar.selectbox("Region Name", regions)

inputs['Rooms'] = st.sidebar.number_input("Rooms", 1, 10, 3)
inputs['Bedroom2'] = st.sidebar.number_input("Bedrooms", 1, 10, 3)
inputs['Bathroom'] = st.sidebar.number_input("Bathrooms", 0, 10, 1)
inputs['Car'] = st.sidebar.number_input("Car Spaces", 0, 10, 1)
inputs['Distance'] = st.sidebar.number_input("Distance from City (km)", 0.0, 50.0, 10.0)
inputs['Postcode'] = st.sidebar.number_input("Postcode", 1000, 3999, 3000)
inputs['BuildingArea'] = st.sidebar.number_input("Building Area (m²)", 30, 1000, 150)
inputs['YearBuilt'] = st.sidebar.number_input("Year Built", 1800, 2025, 2010)
inputs['Propertycount'] = st.sidebar.number_input("Properties in Suburb", 100, 50000, 1000)

# Include Landsize only if model expects it
if "Landsize" in feature_names:
    inputs['Landsize'] = st.sidebar.number_input("Land Size (m²)", 50, 2000, 500)

# ------------------------------------------
# Prediction Section
# ------------------------------------------
st.subheader("📈 House Price Prediction")

def normalize_inputs(df):
    """Clamp unrealistic values to prevent out-of-range errors."""
    df['Distance'] = np.clip(df['Distance'], 0, 50)
    df['BuildingArea'] = np.clip(df['BuildingArea'], 30, 1000)
    if 'Landsize' in df:
        df['Landsize'] = np.clip(df['Landsize'], 50, 2000)
    df['Rooms'] = np.clip(df['Rooms'], 1, 10)
    df['Bathroom'] = np.clip(df['Bathroom'], 0, 10)
    df['Car'] = np.clip(df['Car'], 0, 10)
    return df

if st.button("Predict Price"):
    df_input = pd.DataFrame([inputs])
    df_input = normalize_inputs(df_input)

    # Encode categorical features safely
    for col, le in encoders.items():
        if col in df_input:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                st.warning(f"⚠️ '{df_input[col].values[0]}' not found in training data for '{col}'. Using default.")
                df_input[col] = le.transform([le.classes_[0]])

    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    try:
        prediction = model.predict(df_input)[0]

        # If trained on log(price), apply exponential correction
        if prediction < 0:
            st.warning("⚠️ Raw model output was negative. Adjusting to realistic minimum.")
            prediction = abs(prediction) * 0.75  # simple correction

        prediction = np.maximum(prediction, 50000)  # Ensure min threshold
        st.success(f"💰 Estimated House Price: **₹ {prediction:,.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------
# 💬 AI Chatbot Section
# ------------------------------------------
st.write("---")
st.subheader("💬 AI Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

faq = {
    "what is dss": "A Decision Support System helps make informed, data-driven decisions — here, it predicts house prices.",
    "how prediction works": "The model processes your inputs, encodes categorical data, and estimates a house price using regression.",
    "rooms": "Total number of rooms in the property.",
    "bedroom2": "Number of bedrooms.",
    "bathroom": "Number of bathrooms.",
    "car": "Available parking spaces.",
    "distance": "Distance from Melbourne’s Central Business District in kilometers.",
    "landszie": "Land area in square meters.",
    "buildingarea": "Total constructed floor area of the house.",
    "yearbuilt": "The year the house was built.",
    "suburb": "Local area or neighborhood of the property.",
    "regionname": "Larger region grouping suburbs.",
    "propertycount": "Number of properties in that suburb.",
    "thank you": "You're welcome 😊",
}

def get_bot_response(query):
    query = query.lower().strip()
    for key, ans in faq.items():
        if key in query:
            return ans
    if "price" in query:
        return "The predicted price is based on rooms, building area, and location."
    return "I’m not sure about that — try asking about a specific feature like 'rooms' or 'building area'."

user_query = st.text_input("Ask your question about the model or dataset:", key="chat_input")

if user_query:
    response = get_bot_response(user_query)
    st.session_state.chat_history.append({"sender": "user", "message": user_query})
    st.session_state.chat_history.append({"sender": "bot", "message": response})

for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"<div style='text-align:right; background-color:#1E3A8A; padding:8px; border-radius:10px; margin:5px; color:white;'>🧑‍💻 <b>You:</b> {chat['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; background-color:#3B0764; padding:8px; border-radius:10px; margin:5px; color:white;'>🤖 <b>Bot:</b> {chat['message']}</div>", unsafe_allow_html=True)

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Developed as part of a Decision Support System project using Machine Learning and AI.")
