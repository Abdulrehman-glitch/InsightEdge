from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    project: str
    api_version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_file: str
    feature_count: int
    labels: dict[int, str]


class PredictionRequest(BaseModel):
    vibration_mean: float
    vibration_std: float
    vibration_min: float
    vibration_max: float
    vibration_range: float
    vibration_kurtosis: float

    ax_mean: float
    ax_std: float
    ax_peak_to_peak: float

    ay_mean: float
    ay_std: float
    ay_peak_to_peak: float

    az_mean: float
    az_std: float
    az_peak_to_peak: float

    pressure_mean: float
    pressure_std: float
    pressure_min: float
    pressure_max: float
    pressure_range: float

    temperature_mean: float
    temperature_std: float
    temperature_max: float

    rpm_mean: float
    rpm_std: float
    rpm_range: float


class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_fault: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]