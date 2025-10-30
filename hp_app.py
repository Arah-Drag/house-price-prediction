import streamlit as st
import pandas as pd
import joblib
import datetime

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
try:
    model = joblib.load("pipeline_model.pkl")
except Exception as e:
    st.error(f"❌ Required files not found: {e}")
    st.stop()

# -------------------------------------------------------
# Load Dataset for Dropdown Values
# -------------------------------------------------------
try:
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df.dropna(subset=["location", "society"], inplace=True)
    unique_locations = sorted(df["location"].unique().tolist())
    unique_societies = sorted(df["society"].unique().tolist())
except Exception as e:
    st.warning(f"⚠️ Could not load dataset for dropdowns: {e}")
    unique_locations = ["Whitefield", "Electronic City", "Indira Nagar"]
    unique_societies = ["Prestige Group", "Sobha", "Brigade", "Unknown"]

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="🏠 Bengaluru House Price DSS", layout="wide")
st.title("🏠 Bengaluru House Price Prediction DSS")
st.write("An AI-powered Decision Support System (DSS) using Random Forest to forecast house prices in Bengaluru with explainable insights.")

# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("🔧 Enter Property Details")

# Dropdowns and Inputs
area_type = st.sidebar.selectbox("🏢 Area Type", ["Super built-up  Area", "Plot  Area", "Built-up  Area", "Carpet  Area"])
society = st.sidebar.selectbox("🏘️ Society Name", unique_societies)
location = st.sidebar.selectbox("📍 Location", unique_locations)
total_sqft = st.sidebar.number_input("📏 Total Area (sqft)", min_value=300.0, max_value=10000.0, value=1200.0, step=10.0)
bath = st.sidebar.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.sidebar.number_input("🌿 Number of Balconies", min_value=0, max_value=5, value=1)
bhk = st.sidebar.number_input("🛏️ Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)

# Prepare input
df_input = pd.DataFrame([{
    "area_type": area_type,
    "society": society,
    "location": location,
    "total_sqft": total_sqft,
    "bath": bath,
    "balcony": balcony,
    "BHK": bhk
}])

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
st.subheader("📈 Predicted House Price")

if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Estimated House Price: **₹ {prediction * 100000:,.2f}**")
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
    # --- Core Feature Explanations ---
    "what is bhk": "BHK stands for Bedroom, Hall, and Kitchen — a standard way to describe property size in India.",
    "what is total_sqft": "Total_sqft means the total built-up area of the property in square feet.",
    "what is area_type": "Area Type specifies whether it's a Super built-up Area, Built-up Area, Carpet Area, or Plot Area.",
    "what is society": "Society refers to the residential project or gated community where the property is located.",
    "what is location": "Location represents the neighborhood or area in Bengaluru where the property lies.",
    "what is balcony": "Balcony indicates the number of outdoor spaces attached to the property.",
    "what is bath": "Number of bathrooms available in the house.",
    # --- Model & DSS Explanations ---
    "how is price calculated": "The system uses a trained Random Forest Regression model that analyzes numerical and categorical features like size, area type, society, and location to predict the property price.",
    "which model is used": "We use a Random Forest Regressor — an ensemble learning model that combines multiple decision trees for better accuracy.",
    "what is dss": "A Decision Support System (DSS) is a data-driven system that helps users make informed decisions using AI and analytics.",
    "what is machine learning": "Machine Learning enables computers to learn from data and make predictions — here, it helps estimate house prices.",
    # --- Project Specific FAQs ---
    "what is the purpose of this project": "The project's goal is to develop an AI-based DSS that predicts house prices and assists real estate decision-making in Bengaluru.",
    "what dataset is used": "We used the 'Bengaluru House Data' dataset, containing features like area type, total sqft, bathrooms, balconies, and location.",
    "what algorithm is used": "We implemented Random Forest Regressor, a supervised learning algorithm.",
    "how accurate is the model": "The current model achieved around 50–55% accuracy (R² score).",
    "what can improve accuracy": "Cleaning missing data, encoding categorical variables properly, and including more relevant features can improve performance.",
    "what is preprocessing": "Preprocessing means cleaning, encoding, and scaling data before training the ML model.",
    "thank you": "You're welcome! 😊",
    "who developed this system": "This DSS was developed by Arasu as part of an M.Sc. Computational Statistics and Data Analytics project at Gandhinagar, Gujarat (2022–2024)."
}

def get_bot_response(query):
    query = query.lower()
    for key, answer in faq.items():
        if key in query:
            return answer
    return "🤖 I’m not sure about that. Try asking about features, model, or project details."

user_query = st.text_input("Ask your question:", key="chat_input")

if user_query:
    response = get_bot_response(user_query)
    st.session_state.chat_history.append({"sender": "user", "message": user_query, "time": datetime.datetime.now().strftime("%H:%M:%S")})
    st.session_state.chat_history.append({"sender": "bot", "message": response, "time": datetime.datetime.now().strftime("%H:%M:%S")})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["sender"] == "user":
        st.markdown(f"<div style='text-align:right; background-color:#0f3460; padding:8px; border-radius:10px; margin:5px; color:white;'>🧑‍💻 <b>You:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; background-color:#16213e; padding:8px; border-radius:10px; margin:5px; color:white;'>🤖 <b>Bot:</b> {chat['message']}<br><small>{chat['time']}</small></div>", unsafe_allow_html=True)

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

st.markdown("---")
st.caption("Developed by Arasu | M.Sc. Computational Statistics & Data Analytics | DSS using Machine Learning & AI 🤖")
