from __future__ import annotations

from datetime import datetime
from typing import Union

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    turbine_id:              str   = Field(..., min_length=1)
    measurement_timestamp:   str
    temperature:             float = Field(..., ge=0.0)
    humidity:                float = Field(..., ge=0.0, le=100.0)
    noise_level:             float = Field(..., ge=0.0)

    @field_validator("measurement_timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y%m%d_%H%M%S")
        except ValueError:
            raise ValueError("measurement_timestamp must follow format yyyymmdd_hhmmss")
        return v


class PredictResponse(BaseModel):
    turbine_id:              str
    measurement_timestamp:   str
    response_timestamp:      str
    potential_anomaly:       bool
    probability:             float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    timestamp:     str


class ErrorDetail(BaseModel):
    code:       str
    message:    str
    timestamp:  str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class BatchItemSuccess(BaseModel):
    index:                   int
    turbine_id:              str
    measurement_timestamp:   str
    response_timestamp:      str
    potential_anomaly:       bool
    probability:             float


class BatchItemError(BaseModel):
    index:  int
    error:  dict


class BatchResponse(BaseModel):
    total:      int
    succeeded:  int
    failed:     int
    results:    list[Union[BatchItemSuccess, BatchItemError]]
