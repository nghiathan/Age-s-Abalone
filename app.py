from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from keras.models import load_model
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model and preprocessor
model = load_model("abalone_model.keras")
preprocessor = joblib.load("abalone_preprocessor.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Map numeric sex value to string
        sex_map = {0: 'M', 1: 'F', 2: 'I'}
        sex_str = sex_map.get(data['sex'], 'I') # Default to 'I' if something else is passed

        # Create a DataFrame from the input data with the correct column names
        input_data = pd.DataFrame({
            'Sex': [sex_str],
            'Length': [data['length']],
            'Diameter': [data['diameter']],
            'Height': [data['height']],
            'Whole weight': [data['whole_weight']],
            'Shucked weight': [data['shucked_weight']],
            'Viscera weight': [data['viscera_weight']],
            'Shell weight': [data['shell_weight']]
        })

        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)

        # Make a prediction
        prediction_rings = model.predict(input_processed)
        
        # Calculate age
        predicted_age = float(prediction_rings[0][0]) + 1.5

        return jsonify({"predicted_age": predicted_age})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)