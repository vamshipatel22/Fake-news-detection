import argparse
import joblib
from src.preprocess import clean_text
def predict_text(model_path, texts):
pipeline = joblib.load(model_path)
cleaned = [clean_text(t) for t in texts]
preds = pipeline.predict(cleaned)
probs = pipeline.predict_proba(cleaned) if hasattr(pipeline, 'predict_proba') else None
return list(preds), probs
if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to trained model')
parser.add_argument('--text', required=True, help='Text to classify')
args = parser.parse_args()
pred, probs = predict_text(args.model, [args.text])
label = 'Fake' if pred[0] == 1 else 'Real'
print(f'Prediction: {label}')
if probs is not None:
print('Probabilities:', probs[0].tolist())
