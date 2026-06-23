from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "models" / "fault_classifier.joblib"
FEATURE_COLUMNS_FILE = PROJECT_ROOT / "data" / "models" / "feature_columns.json"

LABEL_NAMES = {
    0: "NORMAL",
    1: "BEARING_WEAR",
    2: "PRESSURE_ANOMALY",
    3: "IMPACT_EVENT",
}

SEVERITY_MAP = {
    "NORMAL": "LOW",
    "BEARING_WEAR": "HIGH",
    "PRESSURE_ANOMALY": "MEDIUM",
    "IMPACT_EVENT": "CRITICAL",
}


def check_required_files() -> None:
    required_files = [
        FEATURE_FILE,
        MODEL_FILE,
        FEATURE_COLUMNS_FILE,
    ]

    missing_files = [file for file in required_files if not file.exists()]

    if missing_files:
        missing_text = "\n".join(str(file) for file in missing_files)
        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{missing_text}\n\n"
            "Run these first:\n"
            "python edge-simulator\\generate_data.py\n"
            "python scripts\\extract_features.py\n"
            "python ml\\train_fault_classifier.py"
        )


def load_model_assets():
    model = joblib.load(MODEL_FILE)

    feature_columns = json.loads(
        FEATURE_COLUMNS_FILE.read_text(encoding="utf-8")
    )

    features_df = pd.read_csv(FEATURE_FILE)

    return model, feature_columns, features_df


def predict_sample_windows(model, feature_columns, features_df: pd.DataFrame) -> None:
    sample_groups = []

    for scenario in sorted(features_df["scenario"].unique()):
        scenario_rows = features_df[features_df["scenario"] == scenario]

        if len(scenario_rows) < 2:
            raise ValueError(
                f"Not enough rows for scenario '{scenario}'. "
                "Expected at least 2 rows per scenario."
            )

        sample_groups.append(
            scenario_rows.sample(n=2, random_state=42)
        )

    sample_df = pd.concat(sample_groups, ignore_index=True)

    X_sample = sample_df[feature_columns]

    predictions = model.predict(X_sample)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_sample)

    print("InsightEdge standalone model prediction test")
    print("=" * 55)
    print()

    for index, row in sample_df.iterrows():
        actual_label = int(row["label"])
        predicted_label = int(predictions[index])

        actual_fault = LABEL_NAMES[actual_label]
        predicted_fault = LABEL_NAMES[predicted_label]

        confidence = None
        if probabilities is not None:
            confidence = probabilities[index][predicted_label]

        print(f"Sample {index + 1}")
        print(f"Scenario:        {row['scenario']}")
        print(f"Actual fault:    {actual_fault}")
        print(f"Predicted fault: {predicted_fault}")
        print(f"Severity:        {SEVERITY_MAP[predicted_fault]}")

        if confidence is not None:
            print(f"Confidence:      {confidence:.4f}")

        if actual_label == predicted_label:
            print("Result:          PASS")
        else:
            print("Result:          FAIL")

        print("-" * 55)


def main() -> None:
    check_required_files()
    model, feature_columns, features_df = load_model_assets()
    predict_sample_windows(model, feature_columns, features_df)


if __name__ == "__main__":
    main()