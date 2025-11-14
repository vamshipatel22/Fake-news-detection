import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
PS = PorterStemmer()
def load_dataset(path: str) -> pd.DataFrame:
df = pd.read_csv(path)
return df
def clean_text(text: str) -> str:
if not isinstance(text, str):
return ""
text = text.lower()
text = re.sub(r"http\S+", "", text)
text = re.sub(r"[^a-z0-9\s]", " ", text)
tokens = text.split()
tokens = [t for t in tokens if t not in STOPWORDS]
tokens = [PS.stem(t) for t in tokens]
return " ".join(tokens)
def prepare_dataframe(df: pd.DataFrame, text_cols=None, label_col='label') -> pd.DataFrame:
if text_cols is None:
# prefer 'text' or 'title' if available
if 'text' in df.columns:
text_cols = ['text']
elif 'title' in df.columns:
text_cols = ['title']
else:
# use first column as text
text_cols = [df.columns[0]]
# combine text columns
df = df.copy()
df['combined_text'] = df[text_cols].astype(str).agg(' '.join, axis=1)
df['clean_text'] = df['combined_text'].apply(clean_text)
if label_col in df.columns:
df = df[[ 'clean_text', label_col ]]
df = df.rename(columns={label_col: 'label'})
else:
df = df[['clean_text']]
return df
