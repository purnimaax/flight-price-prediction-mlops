from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import uvicorn

# 1. Initialize FastAPI
app = FastAPI(
    title="Flight Price Prediction API",
    description="API to predict flight prices using XGBoost",
    version="1.0.0"
)

# 2. Load Artifacts (Global State)
print("Loading model artifacts...")
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
    metadata = pickle.load(open('metadata.pkl', 'rb'))
    
    feature_names = metadata['feature_names']
    numeric_cols = metadata['numeric_cols']
    categorical_cols = metadata['categorical_cols']
    print("✓ Artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    raise e

# 3. Define Input Data Schema (Pydantic)
class FlightInput(BaseModel):
    airline: str
    flight: str
    source_city: str
    departure_time: str
    stops: str        # Note: Keeps as string to match your encoder logic
    arrival_time: str
    destination_city: str
    class_type: str   # 'class' is a reserved keyword in Python, so we use class_type
    duration: float
    days_left: int

    class Config:
        # Example for the Swagger UI
        json_schema_extra = {
            "example": {
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
        }

# 4. Define Prediction Endpoint
@app.post("/predict")
def predict_price(input_data: FlightInput):
    try:
        # Convert Pydantic object to Dict
        data = input_data.dict()
        
        # Rename 'class_type' back to 'class' for the model
        data['class'] = data.pop('class_type')

        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Ensure column order matches training
        df = df[feature_names]

        # A. Encode Categorical Columns
        for col in categorical_cols:
            if col in encoders:
                # Handle unknown categories gracefully (optional improvement)
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Unknown category in column '{col}'")

        # B. Scale Numeric Columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # C. Predict
        prediction = model.predict(df)[0]
        
        return {
            "predicted_price": float(prediction),
            "currency": "INR"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Health Check Endpoint
@app.get("/")
def home():
    return {"message": "Flight Price Prediction API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)