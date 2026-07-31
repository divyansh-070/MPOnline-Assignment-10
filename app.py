"""
app.py
------
Flask REST API for the Heart Disease Prediction model.

Endpoints:
    GET  /          -> health check / usage info
    POST /predict    -> accepts patient details as JSON, returns a prediction

Run locally:
    python app.py
    # then, in another terminal:
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
              "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
              "slope":0,"ca":0,"thal":1}'
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

MODEL_PATH = "model.pkl"
FEATURE_ORDER_PATH = "feature_order.json"

app = Flask(__name__)

# Load model and expected feature order once, at startup.
model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURE_ORDER_PATH):
    with open(FEATURE_ORDER_PATH) as f:
        FEATURE_ORDER = json.load(f)
else:
    # Fallback: standard column order for the Kaggle heart-disease dataset.
    FEATURE_ORDER = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient details as JSON to /predict",
        "web_form": "/ui",
        "expected_fields": FEATURE_ORDER
    })


@app.route("/ui", methods=["GET"])
def ui():
    """Optional browser-based form (templates/index.html) that calls /predict."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    missing_fields = [f for f in FEATURE_ORDER if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required field(s).",
            "missing_fields": missing_fields,
            "expected_fields": FEATURE_ORDER
        }), 400

    try:
        # Build a single-row DataFrame in the exact column order the
        # model was trained on.
        input_row = pd.DataFrame([[data[field] for field in FEATURE_ORDER]],
                                  columns=FEATURE_ORDER)
        pred = model.predict(input_row)[0]
        proba = model.predict_proba(input_row)[0].tolist() if hasattr(model, "predict_proba") else None

        result = {
            "prediction": "Heart Disease Detected" if int(pred) == 1 else "No Heart Disease Detected"
        }

        return jsonify(result)

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input value(s): {str(e)}"}), 400


if __name__ == "__main__":
    # For local development only. Render runs this via Gunicorn (see Procfile).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
