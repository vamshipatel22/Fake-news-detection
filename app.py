from flask import Flask, request, jsonify
import joblib
from src.preprocess import clean_text
app = Flask(__name__)
MODEL_PATH = 'models/model.joblib'
pipeline = joblib.load(MODEL_PATH)
@app.route('/predict', methods=['POST'])
def predict():
body = request.get_json()
if not body or 'text' not in body:
return jsonify({'error': 'No text provided'}), 400
text = body['text']
x = clean_text(text)
pred = pipeline.predict([x])[0]
prob = None
if hasattr(pipeline, 'predict_proba'):
prob = pipeline.predict_proba([x])[0].tolist()
label = 'Fake' if int(pred) == 1 else 'Real'
return jsonify({'label': label, 'probabilities': prob})
if __name__ == '__main__':
app.run(debug=True)
