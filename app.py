from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
'''
 Test CURL response:
 
    No Stroke:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\":[[0.21114288081685926, -0.8157771112861038, 0.5504701870050672, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]}"

    Stroke:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\":[[70, 1, 1, 190, 30, 1, 0, 0, 1, 0, 1, 0, 1]]}"

Index(['age', 'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes'],
      dtype='object')
'''

# Load the trained model
model = joblib.load('stroke_prediction_model.xgb')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']

    # Make prediction
    prediction = model.predict(features)

    # Return results in JSON format
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
