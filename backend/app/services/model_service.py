from pathlib import Path
import json

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


class FaultPredictionService:
    def __init__(self) -> None:
        self.model = None
        self.feature_columns: list[str] = []

    def load(self) -> None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

        if not FEATURE_COLUMNS_FILE.exists():
            raise FileNotFoundError(
                f"Feature columns file not found: {FEATURE_COLUMNS_FILE}"
            )

        self.model = joblib.load(MODEL_FILE)
        self.feature_columns = json.loads(
            FEATURE_COLUMNS_FILE.read_text(encoding="utf-8")
        )

    def is_loaded(self) -> bool:
        return self.model is not None and len(self.feature_columns) > 0

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model.__class__.__name__ if self.model else "Not loaded",
            "model_file": str(MODEL_FILE),
            "feature_count": len(self.feature_columns),
            "labels": LABEL_NAMES,
        }

    def predict(self, features: dict) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model service is not loaded.")

        missing_features = [
            column for column in self.feature_columns
            if column not in features
        ]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        ordered_features = {
            column: features[column]
            for column in self.feature_columns
        }

        input_df = pd.DataFrame([ordered_features])

        predicted_label = int(self.model.predict(input_df)[0])
        predicted_fault = LABEL_NAMES[predicted_label]

        probabilities = {}

        if hasattr(self.model, "predict_proba"):
            probability_values = self.model.predict_proba(input_df)[0]

            for label_id, probability in zip(self.model.classes_, probability_values):
                fault_name = LABEL_NAMES[int(label_id)]
                probabilities[fault_name] = round(float(probability), 4)

            confidence = probabilities[predicted_fault]
        else:
            confidence = 1.0
            probabilities[predicted_fault] = 1.0

        return {
            "predicted_label": predicted_label,
            "predicted_fault": predicted_fault,
            "severity": SEVERITY_MAP[predicted_fault],
            "confidence": confidence,
            "probabilities": probabilities,
        }


prediction_service = FaultPredictionService()