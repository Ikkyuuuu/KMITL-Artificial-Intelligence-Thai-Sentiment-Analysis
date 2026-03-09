import os
import re
import json
import io
from tokenizer import smart_tokenizer
import emoji
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from scipy.sparse import hstack

PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__, template_folder="template", static_folder="static")

stopwords = set(thai_stopwords())
stopwords.discard('ไม่')
stopwords.discard('ดี')

def clean_text_advanced(text):
    text = str(text)
    text = re.sub(r'([ก-๙])\1{2,}', r'\1\1', text)
    text = re.sub(r'http\S+|www\S+', ' <URL> ', text)
    text = re.sub(r'@\S+', ' <USER> ', text)
    text = emoji.replace_emoji(text, replace=' <EMOJI> ')
    text = re.sub(r'[^ก-๙a-zA-Z0-9\s_<>!]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"))
tfidf_word = joblib.load(os.path.join(MODEL_DIR, "tfidf_word.pkl"))
tfidf_char = joblib.load(os.path.join(MODEL_DIR, "tfidf_char.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def predict_sentiment(text):
    cleaned = clean_text_advanced(text)
    X_word = tfidf_word.transform([cleaned])
    X_char = tfidf_char.transform([cleaned])
    X_final = hstack([X_word, X_char])
    
    probs = model.predict_proba(X_final)
    best_index = np.argmax(probs)
    label = le.inverse_transform([best_index])[0]
    confidence = float(np.max(probs)) * 100
    return label, round(confidence, 2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    label, confidence = predict_sentiment(data["prompt"])
    return jsonify({
        "sentiment": label,
        "confidence": confidence
    })

@app.route("/api/upload_json", methods=["POST"])
def upload_json():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files["file"]
    try:
        data = json.loads(file.read().decode("utf-8"))
        if isinstance(data, dict) and "text" in data:
            results = {}
            for key, text in data["text"].items():
                label, _ = predict_sentiment(text)
                results[key] = label
            data["sentiment"] = results
        
        buffer = io.BytesIO()
        buffer.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
        buffer.seek(0)
        return send_file(buffer, mimetype="application/json", as_attachment=True, download_name="result.json")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("--- Thai Sentiment Web App is starting ---")
    print(f"Running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)