import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.preprocess import load_dataset, prepare_dataframe
def evaluate(data_path, model_path):
df = load_dataset(data_path)
df = prepare_dataframe(df)
X = df['clean_text']
y = df['label']
pipeline = joblib.load(model_path)
y_pred = pipeline.predict(X)
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1-score:', f1)
print('\nClassification report:\n')
print(classification_report(y, y_pred))
print('\nConfusion matrix:\n')
print(confusion_matrix(y, y_pred))
if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Path to test CSV')
parser.add_argument('--model', required=True, help='Path to trained model (joblib)')
args = parser.parse_args()
evaluate(args.data, args.model)
