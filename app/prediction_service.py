# app/prediction_service.py
import logging
from PIL import Image

from .image_utils import ImageDecodingError, decode_base64_image
from .model import get_model
from .schemas import PredictRequest, PredictResponse, PredictResult

logger = logging.getLogger(__name__)


def handle_prediction(request: PredictRequest) -> PredictResponse:
    """
    - Decodifica la imagen base64
    - Llama al modelo DenseNet
    - Construye la respuesta
    """
    try:
        image: Image.Image = decode_base64_image(request.image_base64)
        logger.debug("Imagen decodificada correctamente.")

        model = get_model()
        raw_result = model.predict(image)
        logger.debug("Inferencia ejecutada. Resultado: %s", raw_result)

        result = PredictResult(
            prediction=str(raw_result.get("prediction")),
            score=float(raw_result.get("score")),
            extra={k: v for k, v in raw_result.items() if k not in ("prediction", "score")},
        )

        return PredictResponse(success=True, result=result, error=None)

    except ImageDecodingError as img_err:
        logger.warning("Error de decodificación de imagen: %s", img_err)
        return PredictResponse(success=False, result=None, error=str(img_err))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error inesperado durante la predicción.")
        return PredictResponse(
            success=False,
            result=None,
            error=f"Error interno en el backend Python: {exc}",
        )
