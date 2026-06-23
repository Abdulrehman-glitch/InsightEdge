from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "sensor_readings.csv"

RANDOM_SEED = 42
SAMPLES_PER_SCENARIO = 1500
SAMPLE_RATE_HZ = 10

np.random.seed(RANDOM_SEED)


def calculate_rms(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.sqrt((x**2 + y**2 + z**2) / 3)


def generate_normal_data(start_time: datetime) -> pd.DataFrame:
    timestamps = [
        start_time + timedelta(seconds=i / SAMPLE_RATE_HZ)
        for i in range(SAMPLES_PER_SCENARIO)
    ]

    ax = np.random.normal(0.00, 0.035, SAMPLES_PER_SCENARIO)
    ay = np.random.normal(0.00, 0.035, SAMPLES_PER_SCENARIO)
    az = np.random.normal(1.00, 0.040, SAMPLES_PER_SCENARIO)

    pressure = np.random.normal(1013.25, 1.5, SAMPLES_PER_SCENARIO)
    temperature = np.random.normal(28.0, 0.6, SAMPLES_PER_SCENARIO)
    motor_rpm = np.random.normal(1450, 25, SAMPLES_PER_SCENARIO)

    vibration_rms = calculate_rms(ax, ay, az)

    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": "MACHINE_001",
        "scenario": "normal",
        "ax": ax,
        "ay": ay,
        "az": az,
        "vibration_rms": vibration_rms,
        "pressure_hpa": pressure,
        "temperature_c": temperature,
        "motor_rpm": motor_rpm,
        "label": 0,
        "fault_code": "NORMAL",
        "severity": "LOW",
    })


def generate_bearing_wear_data(start_time: datetime) -> pd.DataFrame:
    timestamps = [
        start_time + timedelta(seconds=i / SAMPLE_RATE_HZ)
        for i in range(SAMPLES_PER_SCENARIO)
    ]

    wear_trend = np.linspace(0.00, 0.22, SAMPLES_PER_SCENARIO)

    ax = np.random.normal(0.00, 0.050, SAMPLES_PER_SCENARIO) + wear_trend
    ay = np.random.normal(0.00, 0.055, SAMPLES_PER_SCENARIO) + wear_trend * 0.6
    az = np.random.normal(1.00, 0.060, SAMPLES_PER_SCENARIO)

    periodic_fault = 0.08 * np.sin(np.linspace(0, 80 * np.pi, SAMPLES_PER_SCENARIO))
    ax += periodic_fault

    pressure = np.random.normal(1013.25, 1.8, SAMPLES_PER_SCENARIO)
    temperature = np.random.normal(30.0, 1.0, SAMPLES_PER_SCENARIO) + wear_trend * 6
    motor_rpm = np.random.normal(1430, 40, SAMPLES_PER_SCENARIO)

    vibration_rms = calculate_rms(ax, ay, az)

    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": "MACHINE_001",
        "scenario": "bearing_wear",
        "ax": ax,
        "ay": ay,
        "az": az,
        "vibration_rms": vibration_rms,
        "pressure_hpa": pressure,
        "temperature_c": temperature,
        "motor_rpm": motor_rpm,
        "label": 1,
        "fault_code": "BEARING_WEAR",
        "severity": "HIGH",
    })


def generate_pressure_anomaly_data(start_time: datetime) -> pd.DataFrame:
    timestamps = [
        start_time + timedelta(seconds=i / SAMPLE_RATE_HZ)
        for i in range(SAMPLES_PER_SCENARIO)
    ]

    ax = np.random.normal(0.00, 0.040, SAMPLES_PER_SCENARIO)
    ay = np.random.normal(0.00, 0.040, SAMPLES_PER_SCENARIO)
    az = np.random.normal(1.00, 0.045, SAMPLES_PER_SCENARIO)

    pressure = np.random.normal(1013.25, 1.5, SAMPLES_PER_SCENARIO)
    anomaly_start = SAMPLES_PER_SCENARIO // 3
    anomaly_end = anomaly_start + 500

    pressure[anomaly_start:anomaly_end] += np.linspace(0, 18, anomaly_end - anomaly_start)

    temperature = np.random.normal(29.0, 0.8, SAMPLES_PER_SCENARIO)
    motor_rpm = np.random.normal(1440, 30, SAMPLES_PER_SCENARIO)

    vibration_rms = calculate_rms(ax, ay, az)

    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": "MACHINE_001",
        "scenario": "pressure_anomaly",
        "ax": ax,
        "ay": ay,
        "az": az,
        "vibration_rms": vibration_rms,
        "pressure_hpa": pressure,
        "temperature_c": temperature,
        "motor_rpm": motor_rpm,
        "label": 2,
        "fault_code": "PRESSURE_ANOMALY",
        "severity": "MEDIUM",
    })


def generate_impact_event_data(start_time: datetime) -> pd.DataFrame:
    timestamps = [
        start_time + timedelta(seconds=i / SAMPLE_RATE_HZ)
        for i in range(SAMPLES_PER_SCENARIO)
    ]

    ax = np.random.normal(0.00, 0.040, SAMPLES_PER_SCENARIO)
    ay = np.random.normal(0.00, 0.040, SAMPLES_PER_SCENARIO)
    az = np.random.normal(1.00, 0.045, SAMPLES_PER_SCENARIO)

    impact_indices = [300, 750, 1150]

    for idx in impact_indices:
        ax[idx:idx + 8] += np.random.normal(1.3, 0.25, 8)
        ay[idx:idx + 8] += np.random.normal(0.9, 0.20, 8)
        az[idx:idx + 8] += np.random.normal(1.1, 0.25, 8)

    pressure = np.random.normal(1013.25, 1.6, SAMPLES_PER_SCENARIO)
    temperature = np.random.normal(28.5, 0.7, SAMPLES_PER_SCENARIO)
    motor_rpm = np.random.normal(1450, 28, SAMPLES_PER_SCENARIO)

    vibration_rms = calculate_rms(ax, ay, az)

    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": "MACHINE_001",
        "scenario": "impact_event",
        "ax": ax,
        "ay": ay,
        "az": az,
        "vibration_rms": vibration_rms,
        "pressure_hpa": pressure,
        "temperature_c": temperature,
        "motor_rpm": motor_rpm,
        "label": 3,
        "fault_code": "IMPACT_EVENT",
        "severity": "CRITICAL",
    })


def main() -> None:
    base_time = datetime.now().replace(microsecond=0)

    normal = generate_normal_data(base_time)
    bearing = generate_bearing_wear_data(base_time + timedelta(minutes=3))
    pressure = generate_pressure_anomaly_data(base_time + timedelta(minutes=6))
    impact = generate_impact_event_data(base_time + timedelta(minutes=9))

    dataset = pd.concat(
        [normal, bearing, pressure, impact],
        ignore_index=True
    )

    dataset = dataset.round({
        "ax": 5,
        "ay": 5,
        "az": 5,
        "vibration_rms": 5,
        "pressure_hpa": 2,
        "temperature_c": 2,
        "motor_rpm": 2,
    })

    dataset.to_csv(OUTPUT_FILE, index=False)

    print("InsightEdge sensor dataset generated successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print()
    print("Scenario counts:")
    print(dataset["scenario"].value_counts())
    print()
    print("Fault labels:")
    print(dataset[["scenario", "label", "fault_code", "severity"]].drop_duplicates())


if __name__ == "__main__":
    main()