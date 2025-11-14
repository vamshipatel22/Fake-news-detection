import argparse
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.preprocess import load_dataset, prepare_dataframe
def train(data_path, model_out, vect_out, random_state=42):
df = load_dataset(data_path)
df = prepare_dataframe(df)
X = df['clean_text']
y = df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
clf = LogisticRegression(max_iter=1000)
pipeline = Pipeline([
('vect', vect),
('clf', clf)
])
pipeline.fit(X_train, y_train)
# Save model and vectorizer
os.makedirs(os.path.dirname(model_out), exist_ok=True)
joblib.dump(pipeline, model_out)
# If user wants a separate vectorizer file as well, save pipeline.named_steps['vect']
joblib.dump(pipeline.named_steps['vect'], vect_out)
print(f"Model saved to {model_out}")
print(f"Vectorizer saved to {vect_out}")
if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Path to training CSV')
parser.add_argument('--model_out', default='models/model.joblib', help='Output path for model')
parser.add_argument('--vect_out', default='models/vectorizer.joblib', help='Output path for vectorizer')
args = parser.parse_args()
train(args.data, args.model_out, args.vect_out)
