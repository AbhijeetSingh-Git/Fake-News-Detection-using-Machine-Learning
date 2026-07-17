# Fake News Detection using Machine Learning

This project demonstrates a simple machine learning workflow for identifying whether a news article is likely real or fake. It combines text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier inside a Flask web application.

## Features
- Clean and preprocess news text
- Convert text into TF-IDF features
- Train a Logistic Regression classifier
- Predict whether input text is Real or Fake through a web UI

## Project Structure
- `app.py` — Flask web app and prediction endpoint
- `train.py` — model training and artifact saving
- `templates/index.html` — frontend interface
- `data/news.csv` — sample dataset
- `requirements.txt` — Python dependencies

## Setup
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

## Example Usage
You can paste text such as:
- "A new study shows regular exercise improves heart health."
- "The moon landing was staged in a Hollywood studio."

## GitHub Ready Notes
- Includes a GitHub Actions workflow for basic CI
- Includes a `.gitignore` file for Python projects
- Ready to be pushed to GitHub as a public or private repository
