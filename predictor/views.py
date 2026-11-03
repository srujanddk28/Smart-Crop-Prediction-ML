import json
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render

MODEL = settings.BASE_DIR / "artifacts" / "crop_model.joblib"
METRICS = settings.BASE_DIR / "artifacts" / "metrics.json"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def index(request):
    return render(request, "predictor/index.html", {"model_ready": MODEL.exists()})


def predict_crop(request):
    if request.method != "POST":
        return index(request)

    values = {}
    errors = []
    for field in FEATURES:
        raw_value = request.POST.get(field, "").strip()
        try:
            values[field] = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"Enter a valid number for {field}.")

    if not errors:
        if any(values[k] < 0 for k in ["N", "P", "K", "humidity", "rainfall"]):
            errors.append("N, P, K, humidity and rainfall cannot be negative.")
        if not 0 <= values["ph"] <= 14:
            errors.append("pH must be between 0 and 14.")
        if not -10 <= values["temperature"] <= 60:
            errors.append("Temperature must be between -10°C and 60°C.")
        if values["humidity"] > 100:
            errors.append("Humidity must not exceed 100%.")

    if errors or not MODEL.exists():
        if not MODEL.exists():
            errors.append("Model not trained. Add the CSV dataset and run python scripts/train_model.py.")
        return render(
            request,
            "predictor/index.html",
            {"errors": errors, "values": request.POST, "model_ready": MODEL.exists()},
        )

    try:
        model = joblib.load(MODEL)
        input_frame = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        prediction = str(model.predict(input_frame)[0]).replace("_", " ").title()
        probability = None
        if hasattr(model, "predict_proba"):
            probability = round(float(max(model.predict_proba(input_frame)[0])) * 100, 1)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        return render(
            request,
            "predictor/index.html",
            {"errors": [f"Could not load or use the model: {exc}"], "values": request.POST, "model_ready": True},
        )

    return render(request, "predictor/result.html", {"prediction": prediction, "probability": probability})


def dashboard(request):
    metrics, error = None, None
    if METRICS.exists():
        try:
            metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            error = "Metrics file is unreadable. Please retrain the model."
    else:
        error = "No results yet. Add the dataset and train the model."
    return render(request, "predictor/dashboard.html", {"metrics": metrics, "metrics_error": error})
