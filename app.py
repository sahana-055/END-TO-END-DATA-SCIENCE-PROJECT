from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if _name_ == '_main_':
    app.run(debug=True)