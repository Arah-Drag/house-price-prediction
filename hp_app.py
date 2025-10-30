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
# Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="🏠 Bengaluru House Price DSS", layout="wide")
st.title("🏠 Bengaluru House Price Prediction DSS")
st.write("A Machine Learning-based Decision Support System (DSS) using Random Forest for Bengaluru property valuation.")

# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("🔧 Enter Property Details")

# Inputs
area_type = st.sidebar.selectbox("🏢 Area Type", ["Super built-up  Area", "Plot  Area", "Built-up  Area", "Carpet  Area"])
society = st.sidebar.text_input("🏘️ Society Name", value="Unknown")
location = st.sidebar.text_input("📍 Location", value="Whitefield")
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
    "what is bhk": "BHK stands for Bedroom, Hall, and Kitchen — a common way to describe property size in India.",
    "what is total_sqft": "Total_sqft refers to the total built-up area of the property in square feet.",
    "what is location": "Location indicates the area or locality of the property in Bengaluru.",
    "what is area_type": "Area Type refers to whether the area is Super built-up, Built-up, Carpet, or Plot area.",
    "what is society": "Society represents the residential community or project name where the property is located.",
    "what is balcony": "Balcony is the number of attached outdoor spaces in the house.",
    "how is price calculated": "The model predicts price using Random Forest based on size, location, area type, and other property features.",
    "what is dss": "A Decision Support System (DSS) helps users make informed property decisions using AI.",
    "which model is used": "We use a Random Forest Regressor trained on Bengaluru housing data.",
    "thank you": "You're welcome! 😊"
}

def get_bot_response(query):
    query = query.lower()
    for key, answer in faq.items():
        if key in query:
            return answer
    return "I’m not sure about that. Try asking about features like 'bhk', 'area_type', or 'society'."

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
st.caption("Developed as part of a Decision Support System project using Machine Learning and AI 🤖")
