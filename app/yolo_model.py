# app/yolo_model.py
import logging
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import numpy as np
from io import BytesIO
import base64

from .config import settings

logger = logging.getLogger(__name__)

class YoloSegmentationModel:
    """
    Wrapper para el modelo YOLO de segmentación.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.yolo_seg_model_path
        if not self.model_path:
            raise ValueError("No se ha especificado la ruta del modelo YOLO de segmentación.")
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Cargando modelo YOLO de segmentación desde '%s' ...", self.model_path)
        
        try:
            self._model = YOLO(self.model_path)
            self._model.eval()  # Establecer el modelo en modo evaluación
            logger.info("Modelo YOLO de segmentación cargado correctamente.")
        except Exception as e:
            logger.error("Error al cargar el modelo YOLO: %s", e)
            raise RuntimeError(f"Error al cargar el modelo YOLO: {e}")

    def segment(self, image: Image.Image) -> dict:
        """
        Ejecuta la segmentación sobre una imagen PIL usando el modelo YOLO.

        :param image: Imagen RGB en formato PIL.
        :return: Diccionario con la imagen anotada y metadatos.
        """
        if self._model is None:
            raise RuntimeError("El modelo YOLO aún no ha sido cargado.")

        # Preprocesar la imagen para el modelo
        preprocess = transforms.Compose([
            transforms.Resize((640, 640)),  # Tamaño de entrada del modelo (ajustar si es necesario)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Ajusta la normalización si es necesario
        ])
        
        image_tensor = preprocess(image).unsqueeze(0)  # Agregar dimensión de batch

        with torch.no_grad():  # Desactivar el cálculo de gradientes para predicción
            results = self._model(image_tensor)

        # Revisar los resultados de YOLO
        logger.debug("Resultados de YOLO: %s", results)

        # Procesar los resultados de la segmentación
        annotated_image = self._postprocess_results(results)

        return {
            "annotated_image": annotated_image,
            "num_masks": len(results) if isinstance(results, list) else 0,
            "classes": results.names if hasattr(results, "names") else [],
        }

    def _postprocess_results(self, results) -> Image.Image:
        """
        Convertir los resultados (máscaras) a una imagen anotada.
        """
        # Comprobamos si se generaron máscaras
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks
            logger.debug(f"Máscaras encontradas: {masks.data.shape}")
            
            # Crear una imagen de las máscaras
            annotated = masks.plot()  # Si tu modelo tiene `plot()` o usa OpenCV para dibujar
        else:
            logger.warning("No se encontraron máscaras en los resultados de segmentación.")
            annotated = np.zeros((640, 640, 3), dtype=np.uint8)  # Imagen vacía

        # Convertir de BGR (OpenCV) a RGB (PIL)
        annotated_rgb = annotated[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_rgb.astype(np.uint8))
        return annotated_image


# Singleton sencillo
_yolo_seg_instance: YoloSegmentationModel | None = None


def get_yolo_segmentation_model() -> YoloSegmentationModel:
    global _yolo_seg_instance  # noqa: PLW0603
    if _yolo_seg_instance is None:
        _yolo_seg_instance = YoloSegmentationModel()
    return _yolo_seg_instance

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convierte una imagen PIL a base64 (sin prefijo data:image/...).

    :param image: Imagen PIL.
    :param format: Formato de salida, por defecto PNG.
    :return: Cadena base64.
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded
