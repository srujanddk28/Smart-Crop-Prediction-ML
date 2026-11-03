"""Train a crop classifier and generate honest held-out evaluation outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Crop_recommendation.csv"
OUT = ROOT / "artifacts"
CHARTS = ROOT / "static" / "generated"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset missing: {DATA}. See data/README.md.")
    df = pd.read_csv(DATA)
    required = FEATURES + ["label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(missing))
    df = df[required].dropna()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1)
    X, y = X.loc[valid], df.loc[valid, "label"].astype(str)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise SystemExit("Need at least two crop classes and at least two rows per class.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    dump(model, OUT / "crop_model.joblib")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {"dataset_rows": int(len(X)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "accuracy": float(accuracy_score(y_test, pred)), "macro_precision": float(report["macro avg"]["precision"]), "macro_recall": float(report["macro avg"]["recall"]), "macro_f1": float(report["macro avg"]["f1-score"]), "classes": sorted(y.unique().tolist())}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    counts = y.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, len(counts)*.28)))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Records per crop class"); ax.set_xlabel("Number of records")
    fig.tight_layout(); fig.savefig(CHARTS / "class_distribution.png", dpi=150); plt.close(fig)
    importance = pd.Series(model.named_steps["classifier"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest feature importance"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(CHARTS / "feature_importance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(metrics["classes"])*.45), max(7, len(metrics["classes"])*.4)))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=metrics["classes"], xticks_rotation=90, cmap="Greens", colorbar=False, ax=ax)
    ax.set_title("Confusion matrix · held-out test set")
    fig.tight_layout(); fig.savefig(CHARTS / "confusion_matrix.png", dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
