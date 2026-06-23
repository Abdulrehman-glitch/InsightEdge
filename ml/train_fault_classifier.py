from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

MODEL_FILE = MODEL_DIR / "fault_classifier.joblib"
FEATURE_COLUMNS_FILE = MODEL_DIR / "feature_columns.json"
REPORT_FILE = DOCS_DIR / "fault_classifier_report.md"
CONFUSION_MATRIX_FILE = FIGURES_DIR / "fault_classifier_confusion_matrix.png"

TARGET_COLUMN = "label"

NON_FEATURE_COLUMNS = [
    "machine_id",
    "scenario",
    "fault_code",
    "severity",
    "label",
    "window_start_index",
    "window_end_index",
]

LABEL_NAMES = {
    0: "NORMAL",
    1: "BEARING_WEAR",
    2: "PRESSURE_ANOMALY",
    3: "IMPACT_EVENT",
}


def load_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}. "
            "Run scripts/extract_features.py first."
        )

    return pd.read_csv(FEATURE_FILE)


def prepare_data(df: pd.DataFrame):
    feature_columns = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return X, y, feature_columns


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_model(model_name: str, model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    report = classification_report(
        y_test,
        predictions,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        digits=4,
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": predictions,
        "classification_report": report,
    }


def save_confusion_matrix(best_model_name: str, y_test, predictions) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, xticks_rotation=35)
    ax.set_title(f"Fault Classifier Confusion Matrix - {best_model_name}")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=160)
    plt.close()

    print(f"Saved confusion matrix: {CONFUSION_MATRIX_FILE}")


def save_report(results: list, best_result: dict, feature_columns: list) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# InsightEdge Fault Classifier Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Feature file: `{FEATURE_FILE}`")
    lines.append("- Target: machine fault label")
    lines.append("- Classes:")
    for label_id, label_name in LABEL_NAMES.items():
        lines.append(f"  - `{label_id}` = `{label_name}`")

    lines.append("")
    lines.append("## Feature Leakage Prevention")
    lines.append("")
    lines.append("The following columns were excluded from model training:")
    for col in NON_FEATURE_COLUMNS:
        lines.append(f"- `{col}`")

    lines.append("")
    lines.append("## Sensor-Derived Features Used")
    lines.append("")
    for col in feature_columns:
        lines.append(f"- `{col}`")

    lines.append("")
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 |")
    lines.append("|---|---:|---:|")
    for result in results:
        lines.append(
            f"| {result['model_name']} | "
            f"{result['accuracy']:.4f} | "
            f"{result['macro_f1']:.4f} |"
        )

    lines.append("")
    lines.append("## Selected Model")
    lines.append("")
    lines.append(f"Selected model: **{best_result['model_name']}**")
    lines.append("")
    lines.append("Selection reason: highest macro F1 score on the stratified test split.")
    lines.append("")
    lines.append("## Classification Report")
    lines.append("")
    lines.append("```text")
    lines.append(best_result["classification_report"])
    lines.append("```")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved model report: {REPORT_FILE}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()

    print("Loaded feature dataset.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    X, y, feature_columns = prepare_data(df)

    print("Training feature columns:")
    for col in feature_columns:
        print(f"- {col}")

    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print()

    models = build_models()
    results = []

    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        model.fit(X_train, y_train)

        result = evaluate_model(model_name, model, X_test, y_test)
        results.append(result)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print()

    best_result = max(results, key=lambda item: item["macro_f1"])
    best_model = models[best_result["model_name"]]

    joblib.dump(best_model, MODEL_FILE)

    FEATURE_COLUMNS_FILE.write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )

    save_report(results, best_result, feature_columns)
    save_confusion_matrix(
        best_result["model_name"],
        y_test,
        best_result["predictions"],
    )

    print()
    print("Milestone 4 complete: fault classifier trained and saved.")
    print(f"Best model: {best_result['model_name']}")
    print(f"Model file: {MODEL_FILE}")
    print(f"Feature columns file: {FEATURE_COLUMNS_FILE}")


if __name__ == "__main__":
    main()