import pandas as pd
import pickle
import numpy as np

# Load
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))
meta = pickle.load(open('metadata.pkl', 'rb'))

# Make example
example = pd.DataFrame({
    'airline': ['SpiceJet'],
    'flight': ['SG-8709'],
    'source_city': ['Delhi'],
    'departure_time': ['Evening'],
    'stops': ['zero'],
    'arrival_time': ['Night'],
    'destination_city': ['Mumbai'],
    'class': ['Economy'],
    'duration': [2.17],
    'days_left': [1]
})

# Reorder columns to match training
example = example[meta['feature_names']]

# Step 1: Encode categorical columns FIRST
for col in meta['categorical_cols']:
    example[col] = encoders[col].transform(example[col].astype(str))

# Step 2: Convert to float and scale numeric
example = example.astype(float)
example[meta['numeric_cols']] = scaler.transform(example[meta['numeric_cols']])

# Predict
pred = model.predict(example.values)[0]

print("=" * 60)
print("FLIGHT PRICE PREDICTION")
print("=" * 60)
print(f"\n✓ Predicted Price: ₹{pred:,.0f}")
print("=" * 60)