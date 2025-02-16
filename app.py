from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_recommendation_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Extract soil parameters
    nitrogen = data['nitrogen']
    phosphorus = data['phosphorus']
    potassium = data['potassium']
    ph = data['ph']

    # Make a prediction using the trained model
    prediction = model.predict([[nitrogen, phosphorus, potassium, ph]])
    recommended_crop = prediction[0]

    # Return the result as a JSON response
    return jsonify({
        "recommended_crop": recommended_crop
    })

# Vercel requires this to run the app
if __name__ == '__main__':
    app.run(debug=True)