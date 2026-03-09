import re
import emoji
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

print("Loading data...")
try:
    df = pd.read_json("../../dataset/train_sentiment.json")
    if "text" not in df.columns:
        df = df.transpose().reset_index(drop=True)
except Exception as e:
    print(f"Error loading JSON: {e}")
    exit()

def clean_text_advanced(text):
    text = str(text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' <URL> ', text)
    text = re.sub(r'@\S+', ' <USER> ', text)
    text = text.replace('❤', ' <POS_EMOJI> ').replace('👍', ' <POS_EMOJI> ')
    text = text.replace('👎', ' <NEG_EMOJI> ').replace('😡', ' <NEG_EMOJI> ')
    text = emoji.replace_emoji(text, replace=' <EMOJI> ')
    text = re.sub(r'5{3,}\+?', ' <LAUGH> ', text)
    text = re.sub(r'([ก-๙])\1{2,}', r'\1\1', text)
    text = re.sub(r'[^ก-๙a-zA-Z0-9\s_<>!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text_advanced)

stopwords = set(thai_stopwords())
stopwords.discard('ไม่')
stopwords.discard('ดี') 

def smart_tokenizer(text):
    tokens = word_tokenize(text, engine="newmm")
    result = []
    skip_next = False
    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] == 'ไม่' and i + 1 < len(tokens) and tokens[i+1].strip() != '':
            bound_word = 'ไม่_' + tokens[i+1]
            result.append(bound_word)
            skip_next = True
        else:
            if tokens[i] not in stopwords and tokens[i].strip() != '':
                result.append(tokens[i])
    return result

X = df["clean_text"]
le = LabelEncoder()
y_encoded = le.fit_transform(df["sentiment"])

print(f"Training on all {len(X)} samples...")

print("Vectorizing text (TF-IDF)...")
tfidf_word = TfidfVectorizer(tokenizer=smart_tokenizer, ngram_range=(1, 2), min_df=3, max_features=40000, lowercase=False)
X_word = tfidf_word.fit_transform(X)

tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=20000)
X_char = tfidf_char.fit_transform(X)

X_final = hstack([X_word, X_char])

print("Training Ensemble Model (LGBM + LogReg + Fast SVM)...")
model_lgb = lgb.LGBMClassifier(objective="multiclass", n_estimators=500, learning_rate=0.05, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
model_lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42, n_jobs=-1)
base_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
model_svm_fast = CalibratedClassifierCV(base_svm, cv=3)

ensemble_model = VotingClassifier(estimators=[('lgb', model_lgb), ('lr', model_lr), ('svm', model_svm_fast)], voting='soft', n_jobs=-1)
ensemble_model.fit(X_final, y_encoded)

print("Saving models...")
joblib.dump(ensemble_model, "ensemble_model.pkl")
joblib.dump(tfidf_word, "tfidf_word.pkl")
joblib.dump(tfidf_char, "tfidf_char.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ All processes completed successfully! Final model trained on all data.")