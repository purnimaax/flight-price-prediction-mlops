import streamlit as st
import requests

# -----------------------------------------------------------------------------
# 1. SETUP: POINT TO YOUR RENDER API
# -----------------------------------------------------------------------------

API_URL = "https://flight-price-prediction-uhhi.onrender.com/predict"

st.set_page_config(page_title="Flight Price AI", page_icon="✈️", layout="centered")

# -----------------------------------------------------------------------------
# 2. UI DESIGN
# -----------------------------------------------------------------------------
st.title("✈️ AI Flight Price Predictor")
st.markdown("### Machine Learning Inference System")
st.info(f"Connected to Inference Engine at: `{API_URL}`")

# Create two columns for a clean form
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"])
    flight = st.text_input("Flight Code", "SG-8709")
    source_city = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    departure_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])

with col2:
    arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    destination_city = st.selectbox("Destination City", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    class_type = st.selectbox("Class", ["Economy", "Business"])
    duration = st.number_input("Duration (Hours)", min_value=0.5, max_value=50.0, value=2.5)
    days_left = st.number_input("Days Left", min_value=1, max_value=50, value=5)

# -----------------------------------------------------------------------------
# 3. PREDICTION LOGIC
# -----------------------------------------------------------------------------
if st.button("🚀 Predict Price"):
    payload = {
        "airline": airline,
        "flight": flight,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class_type": class_type,
        "duration": duration,
        "days_left": int(days_left)
    }

    try:
        with st.spinner("Querying Model in the Cloud..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            price = result["predicted_price"]
            st.success(f"💰 Estimated Price: ₹{price:,.0f}")
            st.balloons()  # Fun animation
        else:
            st.error(f"Server Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection Failed! The Render server might be asleep (wait 30s and try again).")
    except Exception as e:
        st.error(f"An error occurred: {e}")