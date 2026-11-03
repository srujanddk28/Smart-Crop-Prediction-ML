"""Train and evaluate the Smart Crop Prediction Random Forest model.

Place the original CSV at data/Crop_recommendation.csv, then run:
    python scripts/train_model.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Crop_recommendation.csv"
OUT = ROOT / "artifacts"
CHARTS = ROOT / "static" / "generated"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def load_dataset():
    if not DATA.is_file():
        raise SystemExit(
            f"Dataset not found: {DATA}\n"
            "Put Crop_recommendation.csv in the data folder. "
            "See data/README.md for the required columns."
        )
    frame = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit("CSV is missing required columns: " + ", ".join(missing))

    frame = frame[required].copy()
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET] = frame[TARGET].astype("string").str.strip().str.lower()
    frame = frame.dropna(subset=required)
    frame = frame[frame[TARGET] != ""]
    if not frame[FEATURES].apply(lambda col: col.map(lambda v: pd.notna(v) and abs(v) != float("inf"))).all().all():
        raise SystemExit("Dataset contains invalid numeric values.")
    if len(frame) == 0:
        raise SystemExit("No usable rows remain after cleaning the CSV.")
    if frame[TARGET].nunique() < 2:
        raise SystemExit("At least two different crop labels are required.")
    if frame[TARGET].value_counts().min() < 2:
        raise SystemExit("Each crop class needs at least two rows for a stratified split.")
    return frame


def main():
    frame = load_dataset()
    X, y = frame[FEATURES], frame[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    dump(model, OUT / "crop_model.joblib")

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    classes = sorted(y.unique().tolist())
    metrics = {
        "model": "RandomForestClassifier",
        "random_state": 42,
        "test_size": 0.20,
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "number_of_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classes": classes,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    counts = y.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, len(counts) * 0.28)))
    counts.plot(kind="barh", ax=ax)
    ax.set(title="Records per crop class", xlabel="Number of records")
    fig.tight_layout()
    fig.savefig(CHARTS / "class_distribution.png", dpi=150)
    plt.close(fig)

    importance = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=FEATURES,
    ).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    importance.plot(kind="barh", ax=ax)
    ax.set(title="Random Forest feature importance", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(max(8, len(classes) * 0.45), max(7, len(classes) * 0.4))
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=classes,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion matrix — held-out test set")
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
