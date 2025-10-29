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

# Numeric Inputs
inputs['Rooms'] = st.sidebar.number_input("Rooms", min_value=1, max_value=10, value=3)
inputs['Bedroom2'] = st.sidebar.number_input("Bedrooms", 1, 10, 3)
inputs['Bathroom'] = st.sidebar.number_input("Bathrooms", 0, 10, 1)
inputs['Car'] = st.sidebar.number_input("Car Spaces", 0, 10, 1)
inputs['Distance'] = st.sidebar.number_input("Distance from City (km)", 0.0, 50.0, 10.0)
inputs['BuildingArea'] = st.sidebar.number_input("Building Area (m²)", 0, 2000, 150)
inputs['YearBuilt'] = st.sidebar.number_input("Year Built", 1800, 2025, 2010)

if "Landsize" in feature_names:
    inputs['Landsize'] = st.sidebar.number_input("Land Size (m²)", 0, 2000, 500)

# ------------------------------------------
# Dropdown Inputs for Categorical Fields
# ------------------------------------------
st.sidebar.markdown("### 🏙️ Select Categorical Details")

# Suburb Options
suburb_options = [
    "Richmond", "Fitzroy", "Carlton", "Brunswick", "Hawthorn",
    "Kew", "Toorak", "South Yarra", "St Kilda", "Footscray"
]
inputs['Suburb'] = st.sidebar.selectbox("Suburb", suburb_options, index=0)

# Type Options
type_options = {"House (h)": "h", "Unit (u)": "u", "Townhouse (t)": "t"}
inputs['Type'] = st.sidebar.selectbox("Property Type", list(type_options.keys()))
inputs['Type'] = type_options[inputs['Type']]

# Method Options
method_options = {
    "S – Sold": "S",
    "SP – Sold Prior": "SP",
    "VB – Vendor Bid": "VB",
    "PI – Passed In": "PI"
}
inputs['Method'] = st.sidebar.selectbox("Sale Method", list(method_options.keys()))
inputs['Method'] = method_options[inputs['Method']]

# Seller Group Options
seller_options = ["Biggin", "Nelson", "Ray White", "Jellis Craig", "Barry Plant", "Harcourts"]
inputs['SellerG'] = st.sidebar.selectbox("Seller Group", seller_options)

# Postcode Options
postcode_options = [3000, 3051, 3065, 3121, 3141, 3182, 3207]
inputs['Postcode'] = st.sidebar.selectbox("Postcode", postcode_options)

# Council Area Options
council_options = ["Yarra", "Melbourne", "Port Phillip", "Boroondara", "Stonnington", "Moreland"]
inputs['CouncilArea'] = st.sidebar.selectbox("Council Area", council_options)

# Region Name Options
region_options = [
    "Northern Metropolitan", "Southern Metropolitan", "Eastern Metropolitan",
    "Western Metropolitan", "Northern Victoria", "Southern Victoria"
]
inputs['Regionname'] = st.sidebar.selectbox("Region Name", region_options)

# Property Count Options
property_options = [500, 1000, 2500, 5000, 10000, 20000]
inputs['Propertycount'] = st.sidebar.selectbox("Properties in Suburb", property_options)

# ------------------------------------------
# Prediction Section
# ------------------------------------------
st.subheader("📈 House Price Prediction")

if st.button("Predict Price"):
    df_input = pd.DataFrame([inputs])

    for col, le in encoders.items():
        if col in df_input:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                st.warning(f"⚠️ Value '{df_input[col].values[0]}' in '{col}' not seen during training. Using default value.")
                df_input[col] = le.transform([le.classes_[0]])

    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Estimated House Price: **₹ {prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------
# 💬 AI Chatbot Section with History + FAQ
# ------------------------------------------
st.write("---")
st.subheader("💬 AI Chatbot Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📚 FAQ Knowledge Base
faq = {
    "what is dss": "A Decision Support System (DSS) helps users make data-driven decisions. Here, it predicts house prices using machine learning.",
    "what is the purpose of this project": "This project predicts house prices using ML models and offers chatbot support for dataset understanding.",
    "what model is used": "We use a Linear Regression model trained on real estate data.",
    "suburb": "Suburb is the local area or neighborhood of the property, affecting its price.",
    "type": "Type specifies the property category — 'h' for house, 'u' for unit, 't' for townhouse.",
    "method": "Method describes how the property was sold — e.g., S (sold), SP (sold prior), VB (vendor bid).",
    "sellerg": "SellerG is the real estate agency that handled the sale.",
    "distance": "Distance shows how far the property is from the city center (CBD).",
    "postcode": "Postcode is the postal code of the property area.",
    "bedroom2": "Bedroom2 indicates the total number of bedrooms.",
    "bathroom": "Bathroom represents how many bathrooms the house has.",
    "car": "Car indicates available parking spaces.",
    "landsize": "Landsize refers to total land area in square meters.",
    "buildingarea": "BuildingArea represents the built-up area of the house.",
    "yearbuilt": "YearBuilt tells the year of construction.",
    "councilarea": "CouncilArea refers to the local government authority.",
    "regionname": "Regionname shows the larger regional area — e.g., 'Northern Metropolitan'.",
    "propertycount": "Propertycount indicates how many properties exist in the same suburb."
}

# Function to generate chatbot response
def get_bot_response(query):
    query = query.lower().strip()
    for key, answer in faq.items():
        if key in query:
            return answer
    if "price" in query:
        return "The predicted price is based on rooms, distance, and area using a trained regression model."
    elif "dataset" in query:
        return "The dataset includes suburb, rooms, area, distance, and sale price information."
    else:
        return "I'm not sure about that. Try asking about a dataset feature like 'rooms', 'landsize', or 'councilarea'."

# User input box
user_query = st.text_input("Ask your question about the model or dataset:", key="chat_input")

# When user sends a message
if user_query:
    response = get_bot_response(user_query)
    st.session_state.chat_history.append({"sender": "user", "message": user_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.session_state.chat_history.append({"sender": "bot", "message": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})

# Display chat history with custom colors
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"<div style='text-align:right; background-color:#1E3A8A; padding:8px; border-radius:10px; margin:5px; color:white;'>🧑‍💻 <b>You:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; background-color:#3B0764; padding:8px; border-radius:10px; margin:5px; color:white;'>🤖 <b>Bot:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Developed as part of a Decision Support System project using Machine Learning and AI.")
