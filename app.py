from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Load dataset
df = pd.read_csv('flightdata.csv')

# Add IsDelayed column if not already present
if 'IsDelayed' not in df.columns and 'ARR_DEL15' in df.columns:
    df['IsDelayed'] = df['ARR_DEL15']

# Drop rows with missing labels
df = df.dropna(subset=['IsDelayed'])

def safe_encode(label, encoder_name):
    encoder_key_map = {
        'carrier': 'unique_carrier',
        'origin': 'origin',
        'destination': 'dest'
    }
    encoder_key = encoder_key_map[encoder_name]
    if encoder_key not in encoders:
        raise ValueError(f"Encoder for '{encoder_key}' not found. Available: {list(encoders.keys())}")
    encoder = encoders[encoder_key]
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        raise ValueError(f"'{label}' not recognized. Please enter a valid {encoder_name.upper()} code.")

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, error_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        flight_number = request.form.get('flightNumber')
        departure_time = request.form.get('departureTime')
        day_of_week = request.form.get('dayOfWeek')
        origin = request.form.get('origin')
        destination = request.form.get('destination')
        carrier = request.form.get('carrier')
        distance = request.form.get('distance')

        if not all([flight_number, departure_time, day_of_week, origin, destination, carrier, distance]):
            raise ValueError("All fields are required.")

        flight_number = int(flight_number)
        day_of_week = int(day_of_week)
        origin = origin.strip().upper()
        destination = destination.strip().upper()
        carrier = carrier.strip().upper()
        distance = float(distance)
        hour = int(departure_time.split(':')[0])

        carrier_encoded = safe_encode(carrier, 'carrier')
        origin_encoded = safe_encode(origin, 'origin')
        destination_encoded = safe_encode(destination, 'destination')

        features = np.array([[hour, day_of_week, flight_number, distance, carrier_encoded, origin_encoded, destination_encoded]])
        prediction = model.predict(features)[0]

        match = df[
            (df['FL_NUM'] == flight_number) &
            (df['ORIGIN'] == origin) &
            (df['DEST'] == destination) &
            (df['UNIQUE_CARRIER'] == carrier)
        ]

        if match.empty:
            return render_template("index.html", error_text="❌ Flight not found in dataset", prediction_text=None)

        delay = match.iloc[0].get('DEP_DELAY', 0)
        cancelled = match.iloc[0].get('CANCELLED', 0)

        if cancelled == 1:
            status = f"❌ Flight {flight_number} is CANCELLED."
        elif prediction == 1:
            status = f"⚠️ Flight {flight_number} is predicted to be DELAYED by approx {int(delay)} minutes."
        else:
            status = f"✅ Flight {flight_number} is predicted to be ON TIME."

        return render_template("index.html", prediction_text=status, error_text=None)

    except ValueError as ve:
        return render_template("index.html", prediction_text=None, error_text=f"❌ Error: {ve}")
    except Exception as e:
        return render_template("index.html", prediction_text=None, error_text=f"❌ Unexpected error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)