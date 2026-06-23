from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "sensor_readings.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

SUMMARY_FILE = PROCESSED_DIR / "dataset_profile.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "scenario",
    "ax",
    "ay",
    "az",
    "vibration_rms",
    "pressure_hpa",
    "temperature_c",
    "motor_rpm",
    "label",
    "fault_code",
    "severity",
]


def load_dataset() -> pd.DataFrame:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_FILE}. "
            "Run edge-simulator/generate_data.py first."
        )

    df = pd.read_csv(RAW_DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    print("Running dataset validation checks...")
    print()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("Scenario counts:")
    print(df["scenario"].value_counts())
    print()

    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    print()

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("Missing values: none")
    else:
        print("Missing values found:")
        print(missing_values)

    print()

    duplicated_rows = df.duplicated().sum()
    print(f"Duplicated rows: {duplicated_rows}")
    print()

    expected_scenarios = {
        "normal",
        "bearing_wear",
        "pressure_anomaly",
        "impact_event",
    }

    actual_scenarios = set(df["scenario"].unique())

    if actual_scenarios != expected_scenarios:
        raise ValueError(
            f"Scenario mismatch. Expected {expected_scenarios}, got {actual_scenarios}"
        )

    print("Dataset validation passed.")


def create_summary(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("scenario")[
            ["ax", "ay", "az", "vibration_rms", "pressure_hpa", "temperature_c", "motor_rpm"]
        ]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    summary.to_csv(SUMMARY_FILE)
    print(f"Dataset profile saved to: {SUMMARY_FILE}")


def plot_scenario_counts(df: pd.DataFrame) -> None:
    counts = df["scenario"].value_counts()

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title("InsightEdge Dataset Scenario Counts")
    plt.xlabel("Scenario")
    plt.ylabel("Number of Readings")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_file = FIGURES_DIR / "scenario_counts.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved graph: {output_file}")


def plot_vibration_boxplot(df: pd.DataFrame) -> None:
    scenarios = ["normal", "bearing_wear", "pressure_anomaly", "impact_event"]
    data = [df[df["scenario"] == scenario]["vibration_rms"] for scenario in scenarios]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=scenarios)
    plt.title("Vibration RMS Distribution by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Vibration RMS")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_file = FIGURES_DIR / "vibration_rms_by_scenario.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved graph: {output_file}")


def plot_pressure_boxplot(df: pd.DataFrame) -> None:
    scenarios = ["normal", "bearing_wear", "pressure_anomaly", "impact_event"]
    data = [df[df["scenario"] == scenario]["pressure_hpa"] for scenario in scenarios]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=scenarios)
    plt.title("Pressure Distribution by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Pressure (hPa)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_file = FIGURES_DIR / "pressure_by_scenario.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved graph: {output_file}")


def plot_time_series(df: pd.DataFrame) -> None:
    sample_df = df.copy()
    sample_df["sample_index"] = range(len(sample_df))

    plt.figure(figsize=(12, 5))

    for scenario in sample_df["scenario"].unique():
        scenario_df = sample_df[sample_df["scenario"] == scenario]
        plt.plot(
            scenario_df["sample_index"],
            scenario_df["vibration_rms"],
            label=scenario,
            linewidth=1,
        )

    plt.title("Vibration RMS Time Series Across Fault Scenarios")
    plt.xlabel("Sample Index")
    plt.ylabel("Vibration RMS")
    plt.legend()
    plt.tight_layout()

    output_file = FIGURES_DIR / "vibration_time_series.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved graph: {output_file}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    validate_dataset(df)
    create_summary(df)

    print()
    print("Generating evidence graphs...")
    plot_scenario_counts(df)
    plot_vibration_boxplot(df)
    plot_pressure_boxplot(df)
    plot_time_series(df)

    print()
    print("Milestone 2 complete: dataset validated and evidence graphs generated.")


if __name__ == "__main__":
    main()