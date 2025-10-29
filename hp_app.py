import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --------------------------------------------------------
# Load Model and Metadata
# --------------------------------------------------------
try:
    model = joblib.load("pipeline_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"❌ Required files not found: {e}")
    st.stop()

# Load dataset (for dropdown options)
try:
    dataset = pd.read_csv("Bengaluru_House_Data.csv")
    dataset.columns = dataset.columns.str.strip().str.lower()
except Exception as e:
    st.warning(f"⚠️ Dataset not found for dropdown population: {e}")
    dataset = pd.DataFrame()

# --------------------------------------------------------
# Page Setup
# --------------------------------------------------------
st.set_page_config(page_title="🏠 Bengaluru House Price DSS", layout="wide")
st.title("🏠 Bengaluru House Price Decision Support System")
st.write("An AI-based tool to estimate property prices in Bengaluru using Machine Learning and interactive chatbot support.")

st.sidebar.header("🧾 Enter Property Details")

# --------------------------------------------------------
# Sidebar Inputs (based on dataset)
# --------------------------------------------------------
inputs = {}

inputs['total_sqft'] = st.sidebar.number_input("Total Square Feet", min_value=200.0, max_value=10000.0, value=1200.0, step=50.0)
inputs['bath'] = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
inputs['balcony'] = st.sidebar.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
inputs['size'] = st.sidebar.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)

if not dataset.empty:
    inputs['location'] = st.sidebar.selectbox("📍 Location", sorted(dataset['location'].dropna().unique().tolist()))
    inputs['area_type'] = st.sidebar.selectbox("🏢 Area Type", sorted(dataset['area_type'].dropna().unique().tolist()))
    inputs['availability'] = st.sidebar.selectbox("📅 Availability", sorted(dataset['availability'].dropna().unique().tolist()))
else:
    inputs['location'] = st.sidebar.text_input("📍 Location", "Whitefield")
    inputs['area_type'] = st.sidebar.text_input("🏢 Area Type", "Super built-up  Area")
    inputs['availability'] = st.sidebar.text_input("📅 Availability", "Ready To Move")

# --------------------------------------------------------
# Prediction Section
# --------------------------------------------------------
st.subheader("📈 Predict House Price")

if st.button("Predict Price"):
    df_input = pd.DataFrame([inputs])

    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Estimated Price: ₹ **{prediction * 1e5:,.2f}**")  # price is in lakhs
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------------
# 💬 Chatbot Section
# --------------------------------------------------------
st.write("---")
st.subheader("💬 AI Chatbot Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

faq = {
    "what is dss": "A Decision Support System (DSS) helps users make better data-driven decisions. Here, it predicts house prices for Bengaluru.",
    "what model is used": "We use a Linear Regression model trained on Bengaluru housing data.",
    "what dataset is used": "The dataset is Bengaluru_House_Data.csv containing details like size, bath, location, area type, and availability.",
    "what is total_sqft": "Total_sqft means the total built-up area of the property in square feet.",
    "what is bath": "Number of bathrooms in the property.",
    "what is balcony": "Number of balconies in the property.",
    "what is size": "Size represents the number of bedrooms (BHK).",
    "what is location": "Location specifies where the house is situated — e.g., Whitefield, Hebbal, etc.",
    "what is area_type": "Area type indicates if it's a super built-up area, carpet area, etc.",
    "what is availability": "Availability shows whether the house is ready to move or under construction.",
    "how prediction works": "The model processes your input through preprocessing and regression pipeline to estimate the house price.",
    "developer": "This DSS was developed by Arasu as part of an academic project.",
    "thank you": "You're welcome! 😊",
}

# Function to get chatbot response
def get_bot_response(query):
    query = query.lower().strip()
    for key, answer in faq.items():
        if key in query:
            return answer

    if "price" in query:
        return "The price prediction is made using a trained linear regression model."
    elif "dataset" in query:
        return "The dataset contains house listings from Bengaluru with features like sqft, BHK, bath, etc."
    elif "feature" in query:
        return "The key features are total_sqft, size (BHK), bath, balcony, location, area_type, and availability."
    else:
        return "I'm not sure about that. Try asking about a specific feature like 'total_sqft' or 'area_type'."

# User input box
user_query = st.text_input("Ask your question about the model or dataset:", key="chat_input")

if user_query:
    response = get_bot_response(user_query)
    st.session_state.chat_history.append({"sender": "user", "message": user_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.session_state.chat_history.append({"sender": "bot", "message": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(
            f"<div style='text-align:right; background-color:#1E3A8A; padding:8px; border-radius:10px; margin:5px; color:white;'>🧑‍💻 <b>You:</b> {chat['message']}<br><small>{chat['time']}</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:left; background-color:#3B0764; padding:8px; border-radius:10px; margin:5px; color:white;'>🤖 <b>Bot:</b> {chat['message']}<br><small>{chat['time']}</small></div>",
            unsafe_allow_html=True,
        )

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

st.markdown("---")
st.caption("Developed as part of an M.Sc. DSS project using Machine Learning and AI 🤖")
