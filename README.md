# Fake News Detection using Machine Learning

![Project Banner](images/hero-banner.svg)

A smart Flask-based web application that predicts whether a piece of text is likely real or fake using machine learning. This project combines text preprocessing, TF-IDF vectorization, and a Logistic Regression classifier to deliver a simple yet effective fake news detection experience.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-orange)

## ✨ Features
- Clean and preprocess news text efficiently
- Convert text into TF-IDF features
- Train a Logistic Regression classifier
- Predict whether input text is Real or Fake through a web interface
- Provide a simple and attractive UI for quick testing

## 📁 Project Structure
- `app.py` — Flask web app and prediction endpoint
- `train.py` — model training and artifact saving
- `templates/index.html` — frontend interface
- `data/news.csv` — sample dataset
- `requirements.txt` — Python dependencies
- `images/hero-banner.svg` — README banner image

## 🚀 Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000

## 🧪 Example Usage
You can paste text such as:
- "A new study shows regular exercise improves heart health."
- "The moon landing was staged in a Hollywood studio."

## 🔥 Why This Project Stands Out
- Easy to understand and beginner-friendly
- Great for learning NLP and machine learning basics
- Ready to be showcased as a portfolio project
- Designed to be simple to run and extend

## 📌 GitHub Ready Notes
- Includes a GitHub Actions workflow for basic CI
- Includes a `.gitignore` file for Python projects
- Ready to be pushed to GitHub as a public or private repository
