from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchItemError,
    BatchItemSuccess,
    BatchResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

MODEL_PATH     = os.getenv("MODEL_PATH", "model/wind_turbine_model.joblib")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10000"))
FEATURES       = ["temperature", "humidity", "noise_level"]

_model = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _error_json(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "timestamp": _now()}},
    )


def _run_inference(req: PredictRequest) -> PredictResponse:
    X = np.array([[req.temperature, req.humidity, req.noise_level]])
    probability = float(_model.predict_proba(X)[0][1])
    return PredictResponse(
        turbine_id=req.turbine_id,
        measurement_timestamp=req.measurement_timestamp,
        response_timestamp=_now(),
        potential_anomaly=probability >= 0.5,
        probability=round(probability, 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as exc:
        print(f"WARNING: model failed to load — {exc}")
        _model = None
    yield


app = FastAPI(
    title="Wind Turbine Anomaly Detection API",
    summary="Real-time failure probability scoring for wind turbine sensor readings.",
    description=(
        "Binary classifier that scores each sensor beat for the probability of a "
        "mechanical or thermal failure, based on generator temperature, relative "
        "humidity, and noise level.\n\n"
        "---\n\n"
        "**Academic context**\n\n"
        "Developed as part of the **From Model to Production** series "
        "(DLBDSMTP01 — Big Data Masterclass) at "
        "**IU International University of Applied Sciences** · Academic Year 2025–2026.\n\n"
        "Academic supervisors: Prof. Dr.-Ing. Anna Androvitsanea · "
        "Prof. Dr. Christian Müller-Kett\n\n"
        "---\n\n"
        "**Integration**\n\n"
        "Designed to run alongside the *Wind Turbine Data Stream Simulator* "
        "(windmill\\_scanner), which calls `POST /api/v1/predict` on every sensor beat. "
        "Readings are stored with `potential_anomaly` and `probability` fields in the "
        "scanner's `sensor_readings` table."
    ),
    version="1.0.0",
    contact={
        "name": "Juan Carlos Laverde",
        "Student ID": "UPS10797707 · Academic Year 2025–2026",
    },
    license_info={
        "name": (
            "IU Internationale Hochschule — Academic Use. "
            "The author donates all rights over this work to IU Internationale Hochschule "
            "for any academic purpose. Source code may be used, adapted, or redistributed "
            "freely for academic and educational purposes."
        ),
    },
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return _error_json("PREDICTION_FAILED", str(exc), 500)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        timestamp=_now(),
    )


# ---------------------------------------------------------------------------
# Single predict
# ---------------------------------------------------------------------------

@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse | JSONResponse:
    if _model is None:
        return _error_json("MODEL_NOT_READY", "Model is not loaded.", 503)
    return _run_inference(req)


# ---------------------------------------------------------------------------
# Batch predict
# ---------------------------------------------------------------------------

@app.post("/api/v1/predict/batch", response_model=BatchResponse)
def predict_batch(body: list[Any] = Body(...)) -> BatchResponse | JSONResponse:
    if _model is None:
        return _error_json("MODEL_NOT_READY", "Model is not loaded.", 503)

    if len(body) > MAX_BATCH_SIZE:
        return _error_json(
            "BATCH_SIZE_EXCEEDED",
            f"Batch size {len(body)} exceeds the maximum of {MAX_BATCH_SIZE}.",
            422,
        )

    results: list[BatchItemSuccess | BatchItemError] = []
    succeeded = 0
    failed = 0

    for i, item in enumerate(body):
        try:
            req = PredictRequest.model_validate(item)
            resp = _run_inference(req)
            results.append(BatchItemSuccess(
                index=i,
                turbine_id=resp.turbine_id,
                measurement_timestamp=resp.measurement_timestamp,
                response_timestamp=resp.response_timestamp,
                potential_anomaly=resp.potential_anomaly,
                probability=resp.probability,
            ))
            succeeded += 1
        except Exception as exc:
            results.append(BatchItemError(
                index=i,
                error={"code": "VALIDATION_ERROR", "message": str(exc)},
            ))
            failed += 1

    return BatchResponse(total=len(body), succeeded=succeeded, failed=failed, results=results)
