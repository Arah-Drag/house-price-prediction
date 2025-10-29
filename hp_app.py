import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

inputs['Rooms'] = st.sidebar.number_input("Rooms", min_value=1, max_value=10, value=3)
inputs['Bedroom2'] = st.sidebar.number_input("Bedrooms", 1, 10, 3)
inputs['Bathroom'] = st.sidebar.number_input("Bathrooms", 0, 10, 1)
inputs['Car'] = st.sidebar.number_input("Car Spaces", 0, 10, 1)
inputs['Type'] = st.sidebar.text_input("Type (h/u/t)", "h")
inputs['Method'] = st.sidebar.text_input("Sale Method (S/SP/VB/PI)", "S")
inputs['SellerG'] = st.sidebar.text_input("Seller Group", "Biggin")
inputs['Distance'] = st.sidebar.number_input("Distance from City (km)", 0.0, 50.0, 10.0)
inputs['Postcode'] = st.sidebar.number_input("Postcode", 1000, 9999, 3000)
inputs['BuildingArea'] = st.sidebar.number_input("Building Area (m²)", 0, 2000, 150)
inputs['YearBuilt'] = st.sidebar.number_input("Year Built", 1800, 2025, 2010)
inputs['CouncilArea'] = st.sidebar.text_input("Council Area", "Yarra")
inputs['Regionname'] = st.sidebar.text_input("Region Name", "Northern Metropolitan")
inputs['Propertycount'] = st.sidebar.number_input("Properties in Suburb", 0, 50000, 1000)
inputs['Suburb'] = st.sidebar.text_input("Suburb", "Richmond")

if "Landsize" in feature_names:
    inputs['Landsize'] = st.sidebar.number_input("Land Size (m²)", 0, 2000, 500)

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
# Feature Importance
# ------------------------------------------
st.subheader("📊 Feature Importance (Coefficients)")
try:
    importance = pd.Series(model.coef_, index=feature_names)
    fig, ax = plt.subplots(figsize=(8, 4))
    importance.abs().sort_values(ascending=False).head(10).plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)
except Exception as e:
    st.info(f"Cannot display feature importance: {e}")

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
    # General DSS and Model Info
    "what is dss": "A Decision Support System (DSS) helps users make better data-driven decisions. Here, it predicts house prices using machine learning.",
    "what is the purpose of this project": "This project predicts house prices using machine learning models and offers chatbot support for dataset understanding.",
    "what model is used": "We use a Linear Regression model trained on real estate data to forecast property prices.",
    "what is the chatbot for": "The chatbot helps users understand how the model, dataset, and features work.",

    # Dataset Feature Explanations
    "suburb": "Suburb is the local area or neighborhood of the property. It strongly influences the price.",
    "address": "Address represents the specific property location.",
    "rooms": "Rooms indicate the total number of rooms in the house (including bedrooms and living areas).",
    "type": "Type specifies the property category — 'h' for house, 'u' for unit, 't' for townhouse.",
    "price": "Price is the target variable — the selling value we aim to predict.",
    "method": "Method describes how the property was sold — e.g., S (sold), SP (sold prior), VB (vendor bid), PI (passed in).",
    "sellerg": "SellerG is the name of the real estate agency that managed the sale.",
    "date": "Date indicates when the property was sold, useful for time-based analysis.",
    "distance": "Distance shows how far the property is from the city center (CBD) in kilometers.",
    "postcode": "Postcode refers to the postal code of the property’s location.",
    "bedroom2": "Bedroom2 shows the total number of bedrooms in the property.",
    "bathroom": "Bathroom indicates the number of bathrooms available.",
    "car": "Car represents the number of parking spaces.",
    "landsize": "Landsize is the total area of the land (in square meters).",
    "buildingarea": "BuildingArea represents the total built-up area of the house (in square meters).",
    "yearbuilt": "YearBuilt tells the year when the property was constructed.",
    "councilarea": "CouncilArea represents the local governing authority of the property.",
    "lattitude": "Lattitude is the north-south geographic coordinate.",
    "longtitude": "Longtitude is the east-west geographic coordinate.",
    "regionname": "Regionname shows the larger regional area — e.g., 'Northern Metropolitan'.",
    "propertycount": "Propertycount shows how many properties are in the same suburb — related to population density.",

    # Chatbot and Prediction
    "how does chatbot work": "The chatbot uses keyword matching to answer dataset and model-related questions.",
    "how prediction works": "The model uses your input details, encodes them, and predicts the house price using Linear Regression.",
    "how to improve accuracy": "Use a larger dataset, retrain the model periodically, or use advanced algorithms like Random Forest or XGBoost.",
    "range": "You can enter Rooms (1–10), Distance (0–50 km), and Building Area (30–1000 sq.m).",
    "developer": "This DSS was developed by Arasu as part of an academic project.",
    "objective": "The DSS helps buyers, sellers, and agents make better real estate decisions.",
    "thank you": "You're welcome! 😊"
}

# Function to generate chatbot response
def get_bot_response(query):
    query = query.lower().strip()

    # Try to match FAQ keywords
    for key, answer in faq.items():
        if key in query:
            return answer

    # Default responses if no match
    if "price" in query:
        return "The predicted price is based on rooms, distance, and area using a trained regression model."
    elif "dataset" in query:
        return "The dataset includes suburb, rooms, area, distance, and sale price information."
    elif "feature" in query or "factor" in query:
        return "The main features are rooms, distance to city, building area, and land size."
    elif "how" in query and "work" in query:
        return "The system cleans data, encodes categories, and predicts price using machine learning."
    else:
        return "I'm not sure about that. Try asking about a dataset feature like 'rooms', 'landsize', or 'councilarea'."

# User input box
user_query = st.text_input("Ask your question about the model or dataset:", key="chat_input")

# When user sends a message
if user_query:
    response = get_bot_response(user_query)

    # Save to chat history
    st.session_state.chat_history.append({"sender": "user", "message": user_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.session_state.chat_history.append({"sender": "bot", "message": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})

# Display chat history
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
