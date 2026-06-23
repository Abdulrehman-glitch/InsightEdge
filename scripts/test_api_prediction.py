from pathlib import Path
import json
import urllib.error
import urllib.request

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
FEATURE_COLUMNS_FILE = PROJECT_ROOT / "data" / "models" / "feature_columns.json"

API_URL = "http://127.0.0.1:8000/api/v1/predict"

LABEL_NAMES = {
    0: "NORMAL",
    1: "BEARING_WEAR",
    2: "PRESSURE_ANOMALY",
    3: "IMPACT_EVENT",
}


def load_assets():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}. "
            "Run scripts/extract_features.py first."
        )

    if not FEATURE_COLUMNS_FILE.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_FILE}. "
            "Run ml/train_fault_classifier.py first."
        )

    features_df = pd.read_csv(FEATURE_FILE)

    feature_columns = json.loads(
        FEATURE_COLUMNS_FILE.read_text(encoding="utf-8")
    )

    return features_df, feature_columns


def send_prediction_request(payload: dict) -> dict:
    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)

    except urllib.error.URLError as error:
        raise ConnectionError(
            "Could not connect to InsightEdge API. "
            "Make sure uvicorn is running on http://127.0.0.1:8000"
        ) from error


def test_predictions(features_df: pd.DataFrame, feature_columns: list[str]) -> None:
    sample_rows = []

    for scenario in sorted(features_df["scenario"].unique()):
        scenario_df = features_df[features_df["scenario"] == scenario]
        sample_rows.append(scenario_df.iloc[0])

    print("InsightEdge API prediction test")
    print("=" * 60)
    print()

    passed = 0

    for index, row in enumerate(sample_rows, start=1):
        payload = {
            column: float(row[column])
            for column in feature_columns
        }

        response = send_prediction_request(payload)

        actual_label = int(row["label"])
        actual_fault = LABEL_NAMES[actual_label]
        predicted_fault = response["predicted_fault"]
        confidence = response["confidence"]

        result = "PASS" if actual_fault == predicted_fault else "FAIL"

        if result == "PASS":
            passed += 1

        print(f"Sample {index}")
        print(f"Scenario:        {row['scenario']}")
        print(f"Actual fault:    {actual_fault}")
        print(f"Predicted fault: {predicted_fault}")
        print(f"Severity:        {response['severity']}")
        print(f"Confidence:      {confidence:.4f}")
        print(f"Result:          {result}")
        print(f"Probabilities:   {response['probabilities']}")
        print("-" * 60)

    print()
    print(f"API prediction test complete: {passed}/{len(sample_rows)} passed")

    if passed != len(sample_rows):
        raise AssertionError("One or more API predictions failed.")


def main() -> None:
    features_df, feature_columns = load_assets()
    test_predictions(features_df, feature_columns)


if __name__ == "__main__":
    main()