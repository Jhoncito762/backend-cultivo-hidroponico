# app/routes.py
import logging

from flask import Blueprint, jsonify, request
from .yolo_service import handle_yolo_segmentation  
from .config import settings
from .prediction_service import handle_prediction
from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationError,
    YoloSegmentResponse
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health_check():
    """
    Endpoint de salud.
    Lo puede usar Java, Kubernetes, etc.
    """
    health = HealthResponse(status="ok", app_name=settings.app_name)
    return jsonify(health.model_dump()), 200


@api_bp.post("/predict")
def predict():
    """
    Endpoint principal de predicción.
    Espera JSON:
    {
      "image_base64": "data:image/jpeg;base64,...."
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON inválido recibido: %s", exc)
        error_response = PredictResponse(
            success=False,
            result=None,
            error="Body JSON inválido.",
        )
        return jsonify(error_response.model_dump()), 400

    try:
        req = PredictRequest.model_validate(data)
    except ValidationError as ve:
        logger.warning("Error de validación: %s", ve)
        error_response = PredictResponse(
            success=False,
            result=None,
            error="Datos de entrada inválidos.",
        )
        return jsonify(error_response.model_dump()), 400

    response = handle_prediction(req)

    status_code = 200 if response.success else 400
    return jsonify(response.model_dump()), status_code


@api_bp.post("/yolo/segment")
def yolo_segment():
    """
    Segmentación con YOLO11.
    Espera JSON:
    {
      "image_base64": "..."
    }
    Devuelve:
    {
      "success": true,
      "segmented_image_base64": "...",
      "num_masks": 3,
      "error": null
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON inválido recibido en /yolo/segment: %s", exc)
        error_response = YoloSegmentResponse(
            success=False,
            segmented_image_base64=None,
            num_masks=None,
            error="Body JSON inválido.",
        )
        return jsonify(error_response.dict()), 400

    try:
        req = PredictRequest(**data)
    except ValidationError as ve:
        logger.warning("Error de validación en /yolo/segment: %s", ve)
        error_response = YoloSegmentResponse(
            success=False,
            segmented_image_base64=None,
            num_masks=None,
            error="Datos de entrada inválidos.",
        )
        return jsonify(error_response.dict()), 400
    
    try:
        from .yoloXD import YoloKhe
        segmented_base64 = YoloKhe.run_inference(req.image_base64)
        
        if segmented_base64:
            response = YoloSegmentResponse(
                success=True,
                segmented_image_base64=segmented_base64,
                num_masks=None,  # Puedes modificar YoloKhe para retornar también el número de máscaras
                error=None
            )
            return jsonify(response.dict()), 200
        else:
            response = YoloSegmentResponse(
                success=False,
                segmented_image_base64=None,
                num_masks=None,
                error="No se pudo procesar la imagen"
            )
            return jsonify(response.dict()), 400
            
    except Exception as exc:
        logger.exception("Error durante la segmentación YOLO: %s", exc)
        error_response = YoloSegmentResponse(
            success=False,
            segmented_image_base64=None,
            num_masks=None,
            error=f"Error interno: {str(exc)}"
        )
        return jsonify(error_response.dict()), 500