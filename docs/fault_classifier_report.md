# InsightEdge Fault Classifier Report

## Dataset

- Feature file: `C:\InsightEdge\data\processed\features.csv`
- Target: machine fault label
- Classes:
  - `0` = `NORMAL`
  - `1` = `BEARING_WEAR`
  - `2` = `PRESSURE_ANOMALY`
  - `3` = `IMPACT_EVENT`

## Feature Leakage Prevention

The following columns were excluded from model training:
- `machine_id`
- `scenario`
- `fault_code`
- `severity`
- `label`
- `window_start_index`
- `window_end_index`

## Sensor-Derived Features Used

- `vibration_mean`
- `vibration_std`
- `vibration_min`
- `vibration_max`
- `vibration_range`
- `vibration_kurtosis`
- `ax_mean`
- `ax_std`
- `ax_peak_to_peak`
- `ay_mean`
- `ay_std`
- `ay_peak_to_peak`
- `az_mean`
- `az_std`
- `az_peak_to_peak`
- `pressure_mean`
- `pressure_std`
- `pressure_min`
- `pressure_max`
- `pressure_range`
- `temperature_mean`
- `temperature_std`
- `temperature_max`
- `rpm_mean`
- `rpm_std`
- `rpm_range`

## Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| logistic_regression | 0.9577 | 0.9575 |
| random_forest | 1.0000 | 1.0000 |

## Selected Model

Selected model: **random_forest**

Selection reason: highest macro F1 score on the stratified test split.

## Classification Report

```text
                  precision    recall  f1-score   support

          NORMAL     1.0000    1.0000    1.0000        18
    BEARING_WEAR     1.0000    1.0000    1.0000        17
PRESSURE_ANOMALY     1.0000    1.0000    1.0000        18
    IMPACT_EVENT     1.0000    1.0000    1.0000        18

        accuracy                         1.0000        71
       macro avg     1.0000    1.0000    1.0000        71
    weighted avg     1.0000    1.0000    1.0000        71

```