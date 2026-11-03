# Smart Crop Prediction using Machine Learning

A Django web application that recommends a crop using soil nutrient and weather readings. The project includes a Random Forest training pipeline, held-out evaluation metrics and visual reports.

## Features

- Crop recommendation from nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH and rainfall.
- Input validation with clear error messages.
- Random Forest model training with a stratified train/test split.
- Accuracy, macro precision, macro recall and macro F1 calculated from held-out test data.
- Dashboard charts for crop distribution, feature importance and confusion matrix.
- Django automated tests and GitHub Actions checks.

## Dataset required

The dataset is intentionally not included in this repository. Obtain a dataset you are permitted to use and place it at:

`data/Crop_recommendation.csv`

The CSV must contain these exact column names:

`N,P,K,temperature,humidity,ph,rainfall,label`

Each row should contain numeric values for the seven input features and a crop name in `label`. Do not paste a partial dataset or use fabricated rows for model training. See [data/README.md](data/README.md).

## Run locally on Windows

1. Install Python 3.10 or newer.
2. Download or clone this repository.
3. Open Command Prompt in the project folder.
4. Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

5. Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Put the complete CSV at `data\Crop_recommendation.csv`.
7. Train the model and generate evaluation files:

```bat
python scripts/train_model.py
```

8. Run Django checks and tests:

```bat
python manage.py check
python manage.py test
```

9. Start the website:

```bat
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/ in your browser.

## Run on macOS or Linux

Use `python3 -m venv .venv`, activate with `source .venv/bin/activate`, then follow the same pip, training and Django commands.

## Generated files

Training creates local files (not committed by default):

- `artifacts/crop_model.joblib` — trained model used by the web app.
- `artifacts/metrics.json` — metrics from the held-out test set.
- `static/generated/class_distribution.png`
- `static/generated/feature_importance.png`
- `static/generated/confusion_matrix.png`

The model uses a fixed random seed for repeatable splitting and training. Reported performance depends on the dataset and should not be interpreted as a guarantee of crop yield or suitability.

## Project structure

- `crop_project/` — Django project settings and URLs.
- `predictor/` — input validation, prediction views and tests.
- `templates/predictor/` — web pages.
- `scripts/train_model.py` — dataset validation, training and evaluation.
- `data/` — place the permitted CSV dataset here.
- `artifacts/` — locally generated model and metrics.

## Important

This is an educational decision-support project, not a substitute for local agricultural advice or a soil test. For deployment, set a secure secret key, `DEBUG=False`, configure `ALLOWED_HOSTS`, and use a production-ready server.
