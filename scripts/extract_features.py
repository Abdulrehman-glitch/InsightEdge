from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "sensor_readings.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_FILE = PROCESSED_DIR / "features.csv"

WINDOW_SIZE = 50
STEP_SIZE = 25


def calculate_kurtosis(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return 0.0

    return float(np.mean(((arr - mean) / std) ** 4))


def most_common_value(series: pd.Series):
    return series.mode().iloc[0]


def extract_window_features(window: pd.DataFrame) -> dict:
    vibration = window["vibration_rms"]
    pressure = window["pressure_hpa"]
    temperature = window["temperature_c"]
    rpm = window["motor_rpm"]

    ax = window["ax"]
    ay = window["ay"]
    az = window["az"]

    return {
        "machine_id": most_common_value(window["machine_id"]),
        "scenario": most_common_value(window["scenario"]),
        "fault_code": most_common_value(window["fault_code"]),
        "severity": most_common_value(window["severity"]),
        "label": int(most_common_value(window["label"])),

        "vibration_mean": vibration.mean(),
        "vibration_std": vibration.std(),
        "vibration_min": vibration.min(),
        "vibration_max": vibration.max(),
        "vibration_range": vibration.max() - vibration.min(),
        "vibration_kurtosis": calculate_kurtosis(vibration),

        "ax_mean": ax.mean(),
        "ax_std": ax.std(),
        "ax_peak_to_peak": ax.max() - ax.min(),

        "ay_mean": ay.mean(),
        "ay_std": ay.std(),
        "ay_peak_to_peak": ay.max() - ay.min(),

        "az_mean": az.mean(),
        "az_std": az.std(),
        "az_peak_to_peak": az.max() - az.min(),

        "pressure_mean": pressure.mean(),
        "pressure_std": pressure.std(),
        "pressure_min": pressure.min(),
        "pressure_max": pressure.max(),
        "pressure_range": pressure.max() - pressure.min(),

        "temperature_mean": temperature.mean(),
        "temperature_std": temperature.std(),
        "temperature_max": temperature.max(),

        "rpm_mean": rpm.mean(),
        "rpm_std": rpm.std(),
        "rpm_range": rpm.max() - rpm.min(),
    }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for scenario in df["scenario"].unique():
        scenario_df = df[df["scenario"] == scenario].reset_index(drop=True)

        for start in range(0, len(scenario_df) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            window = scenario_df.iloc[start:end]

            features = extract_window_features(window)
            features["window_start_index"] = start
            features["window_end_index"] = end - 1
            feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    numeric_columns = features_df.select_dtypes(include=["float64", "float32"]).columns
    features_df[numeric_columns] = features_df[numeric_columns].round(5)

    return features_df


def main() -> None:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_FILE}. "
            "Run edge-simulator/generate_data.py first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_FILE)
    features_df = extract_features(df)

    features_df.to_csv(FEATURE_FILE, index=False)

    print("InsightEdge feature extraction complete.")
    print(f"Input rows: {len(df)}")
    print(f"Feature windows: {len(features_df)}")
    print(f"Output file: {FEATURE_FILE}")
    print()
    print("Feature label counts:")
    print(features_df["scenario"].value_counts())
    print()
    print("Feature columns:")
    print(list(features_df.columns))


if __name__ == "__main__":
    main()