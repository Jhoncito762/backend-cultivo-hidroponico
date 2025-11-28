# app/yolo_service.py
import logging
from PIL import Image

from .image_utils import ImageDecodingError, decode_base64_image, encode_image_to_base64
from .schemas import PredictRequest, YoloSegmentResponse
from .yolo_model import get_yolo_segmentation_model

logger = logging.getLogger(__name__)


def handle_yolo_segmentation(request: PredictRequest) -> YoloSegmentResponse:
    """
    - Decodifica la imagen base64
    - Ejecuta YOLO de segmentación
    - Convierte la imagen anotada a base64
    """
    try:
        image: Image.Image = decode_base64_image(request.image_base64)
        logger.debug("Imagen decodificada correctamente para YOLO.")

        model = get_yolo_segmentation_model()
        result = model.segment(image)
        annotated_image: Image.Image = result["annotated_image"]
        num_masks = result.get("num_masks")

        # Convertimos la imagen anotada a base64
        img_b64 = encode_image_to_base64(annotated_image, format="PNG")

        return YoloSegmentResponse(
            success=True,
            segmented_image_base64=img_b64,
            num_masks=num_masks,
            error=None,
        )

    except ImageDecodingError as img_err:
        logger.warning("Error de decodificación de imagen (YOLO): %s", img_err)
        return YoloSegmentResponse(
            success=False,
            segmented_image_base64=None,
            num_masks=None,
            error=str(img_err),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado durante la segmentación YOLO.")
        return YoloSegmentResponse(
            success=False,
            segmented_image_base64=None,
            num_masks=None,
            error=f"Error interno en el backend YOLO: {exc}",
        )
