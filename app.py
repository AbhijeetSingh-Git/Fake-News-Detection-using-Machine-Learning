from flask import Flask, render_template, request, jsonify
import os
import pickle
import re
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load model and vectorizer if available
model = None
vectorizer = None

if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    article = data.get("article", "")

    if not article.strip():
        return jsonify({"error": "Please enter an article."}), 400

    if model is None or vectorizer is None:
        return jsonify({"error": "Model is not trained yet. Please run train.py first."}), 500

    cleaned = preprocess_text(article)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])

    label = "Real" if prediction == 0 else "Fake"
    return jsonify({
        "prediction": label,
        "confidence": round(probability * 100, 2),
        "probability_real": round((1 - probability) * 100, 2),
        "probability_fake": round(probability * 100, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
