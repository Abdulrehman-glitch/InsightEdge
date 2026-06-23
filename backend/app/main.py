from fastapi import FastAPI, HTTPException

from backend.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.model_service import prediction_service


app = FastAPI(
    title="InsightEdge API",
    description="Backend API for Industrial AI Twin fault prediction.",
    version="0.1.0",
)


@app.on_event("startup")
def startup_event() -> None:
    prediction_service.load()


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "message": "InsightEdge API is running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        project="InsightEdge",
        api_version="0.1.0",
        model_loaded=prediction_service.is_loaded(),
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["ML"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**prediction_service.get_model_info())


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["ML"])
def predict_fault(request: PredictionRequest) -> PredictionResponse:
    try:
        result = prediction_service.predict(request.model_dump())
        return PredictionResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))