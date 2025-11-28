# app/schemas.py
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    app_name: str = Field(..., example="AI Python Flask Backend")


class PredictRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Imagen codificada en base64 (con o sin prefijo data:image/...)",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
    )


class PredictResult(BaseModel):
    prediction: str = Field(..., example="class_1")
    score: float = Field(..., example=0.95)
    extra: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    success: bool = Field(..., example=True)
    result: Optional[PredictResult] = None
    error: Optional[str] = None

class YoloSegmentResponse(BaseModel):
    success: bool = Field(..., example=True)
    segmented_image_base64: Optional[str] = Field(
        default=None,
        description="Imagen segmentada (anotada) codificada en base64",
    )
    num_masks: Optional[int] = Field(default=None, example=3)
    error: Optional[str] = None


__all__ = [
    "HealthResponse",
    "PredictRequest",
    "PredictResult",
    "PredictResponse",
    "ValidationError",
    "YoloSegmentResponse"
]
