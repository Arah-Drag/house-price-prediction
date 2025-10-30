# hp_app.py
import streamlit as st
import pandas as pd
import joblib
import datetime

# -------------------------------------------------------
# Load Model and Encoder
# -------------------------------------------------------
try:
    model = joblib.load("pipeline_model.pkl")
    le = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"❌ Required files not found: {e}")
    st.stop()

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="🏠 Bengaluru House Price DSS", layout="wide")
st.title("🏠 Bengaluru House Price Prediction DSS")
st.write("An intelligent Decision Support System (DSS) using Random Forest for accurate property price forecasting in Bengaluru.")

# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("🔧 Enter Property Details")

# Dropdown for location
location_names = list(le.classes_)
location = st.sidebar.selectbox("📍 Location", location_names)

total_sqft = st.sidebar.number_input("📏 Total Area (sqft)", min_value=300.0, max_value=10000.0, value=1200.0, step=10.0)
bath = st.sidebar.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.sidebar.number_input("🛏️ Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)

inputs = {
    "location": location,
    "total_sqft": total_sqft,
    "bath": bath,
    "BHK": bhk,
    "Area Type": area_type,
    "Availability": availability,
    "Size": size,
    "Society": society 
}

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
st.subheader("📈 Predicted House Price")

if st.button("🔍 Predict Price"):
    try:
        # Encode location
        location_encoded = le.transform([inputs["location"]])[0]

        # Create input DataFrame
        df_input = pd.DataFrame([{
            "location": location_encoded,
            "total_sqft": inputs["total_sqft"],
            "bath": inputs["bath"],
            "BHK": inputs["BHK"]
        }])

        # Predict
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Estimated House Price: **₹ {prediction * 100000:,.2f}**")  # price in ₹
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------------------------
# 💬 Chatbot Section
# -------------------------------------------------------
st.write("---")
st.subheader("💬 AI Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

faq = {
    "what is bhk": "BHK stands for Bedroom, Hall, and Kitchen — a common way to describe property size in India.",
    "what is total_sqft": "Total_sqft refers to the total built-up area of the property in square feet.",
    "what is location": "Location indicates the area or locality of the property in Bengaluru.",
    "how is price calculated": "Price is calculated using features like total area, number of bedrooms, bathrooms, and location using a Random Forest model.",
    "what is dss": "A Decision Support System (DSS) helps make informed decisions using data and AI.",
    "which model is used": "We use a Random Forest Regressor trained on Bengaluru housing data.",
    "thank you": "You're welcome! 😊"
}

def get_bot_response(query):
    query = query.lower()
    for key, answer in faq.items():
        if key in query:
            return answer
    return "I’m not sure about that. Try asking about features like 'bhk', 'total_sqft', or 'location'."

user_query = st.text_input("Ask your question:", key="chat_input")

if user_query:
    response = get_bot_response(user_query)
    st.session_state.chat_history.append({"sender": "user", "message": user_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.session_state.chat_history.append({"sender": "bot", "message": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"<div style='text-align:right; background-color:#1E3A8A; padding:8px; border-radius:10px; margin:5px; color:white;'>🧑‍💻 <b>You:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; background-color:#3B0764; padding:8px; border-radius:10px; margin:5px; color:white;'>🤖 <b>Bot:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

st.markdown("---")
st.caption("Developed as part of a Decision Support System project using Machine Learning and AI 🤖")
