# Smart Crop Prediction using Machine Learning

A Django web app that recommends a crop from soil nutrients and weather readings. It includes a Random Forest training script, evaluation metrics and charts.

## Features
- Enter N, P, K, temperature, humidity, soil pH and rainfall.
- Get a crop recommendation from a trained model.
- View test-set accuracy, macro precision/recall/F1 and charts after training.

## Dataset
Add a permitted CSV file at `data/Crop_recommendation.csv` with columns: `N,P,K,temperature,humidity,ph,rainfall,label`. The dataset is not bundled; check its source and license before using it.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python manage.py migrate
python manage.py runserver
```
Open http://127.0.0.1:8000/. Train the model after placing the CSV. Without a model, the app explains what setup is missing.

## Generated outputs
Training creates `artifacts/crop_model.joblib`, `artifacts/metrics.json`, and charts under `static/generated/`: crop class distribution, feature importance and confusion matrix. Metrics are computed from a stratified held-out test set, never hard-coded.

## Structure
- `crop_project/`: Django configuration
- `predictor/`: views and prediction logic
- `templates/predictor/`: pages
- `scripts/train_model.py`: training and evaluation
- `data/`: local dataset (not committed)
- `artifacts/`: local trained model and metrics (not committed)

Educational decision support only. Validate recommendations with local agricultural experts and soil-test results. A model confidence score is not a yield guarantee. For production, configure a strong secret key, DEBUG=False, allowed hosts and a production server.
