from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    """Test the root endpoint returns 200 OK"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Flight Price Prediction API is running!"}

def test_predict():
    """Test the prediction endpoint with valid data"""
    payload = {
        "airline": "SpiceJet",
        "flight": "SG-8709",
        "source_city": "Delhi",
        "departure_time": "Evening",
        "stops": "zero",
        "arrival_time": "Night",
        "destination_city": "Mumbai",
        "class_type": "Economy",
        "duration": 2.17,
        "days_left": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()