# app/model.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore

from .config import settings

logger = logging.getLogger(__name__)

# Tamaño típico de entrada para DenseNet121 (puedes cambiarlo si tu modelo usa otro)
IMG_SIZE = 128

# Si tienes nombres de clases específicos, ponlos aquí:
# Por ejemplo, si es binario: ["sano", "enfermo"]
CLASS_NAMES: list[str] = ["Harvest Stage", "Seedling Stage", "Vegetative Stage"]  # <-- cámbialos según tu caso


class AIModel:
    """
    Wrapper del modelo de IA basado en Keras (DenseNet121).
    Se encarga de:
    - Cargar el modelo desde disco
    - Preprocesar la imagen
    - Ejecutar la predicción
    - Formatear el resultado
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.model_path
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Carga el modelo Keras desde el archivo .keras.
        """
        if not self.model_path:
            raise ValueError("No se ha especificado la ruta del modelo (model_path).")

        logger.info("Cargando modelo Keras desde '%s' ...", self.model_path)
        start = time.time()

        # Carga del modelo
        self._model = load_model(self.model_path)

        elapsed = time.time() - start
        logger.info("Modelo cargado correctamente en %.3f segundos", elapsed)

    @staticmethod
    def _preprocess_image(image: Image.Image) -> np.ndarray:
        """
        Preprocesa la imagen PIL para que sea compatible con DenseNet.

        - Redimensiona a IMG_SIZE x IMG_SIZE
        - Convierte a numpy array
        - Escala a [0, 1]
        - Añade dimensión batch

        Ajusta aquí si tu modelo usa otro tipo de preprocesamiento.
        """
        # Redimensionar
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))

        # PIL -> numpy (H, W, C)
        img_array = np.array(image_resized, dtype=np.float32)

        # Escalar a [0, 1] (si tu modelo usó otra normalización, cámbialo)
        img_array /= 255.0

        # Añadir dimensión batch: (1, H, W, C)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Ejecuta la inferencia sobre una imagen PIL.

        :param image: Imagen en formato PIL (RGB).
        :return: Diccionario con prediction, score y extra.
        """
        if self._model is None:
            raise RuntimeError("El modelo aún no ha sido cargado.")

        # Preprocesamiento
        input_batch = self._preprocess_image(image)
        print(input_batch.shape)
        start = time.time()

        # Predicción
        preds = self._model.predict(input_batch)
        elapsed_ms = int((time.time() - start) * 1000)

        # Asegurarnos de que sea un array 1D
        preds = np.array(preds).squeeze()

        # Lógica para interpretar salida
        # Si es binario (un solo valor)
        if preds.ndim == 0 or preds.shape == () or preds.shape == (1,):
            score = float(preds)
            # Umbral 0.5 típico
            idx = 1 if score >= 0.5 else 0
            label = (
                CLASS_NAMES[idx]
                if len(CLASS_NAMES) >= 2
                else ("positive" if idx == 1 else "negative")
            )
        else:
            # Multiclase: vector de probabilidades
            idx = int(np.argmax(preds))
            score = float(preds[idx])
            if len(CLASS_NAMES) == preds.shape[0]:
                label = CLASS_NAMES[idx]
            else:
                label = f"class_{idx}"

        return {
            "prediction": label,
            "score": score,
            "tiempo_ms": elapsed_ms,
            "raw_output": preds.tolist(),
            "class_index": idx,
        }


# Instancia global (lazy singleton)
_model_instance: AIModel | None = None


def get_model() -> AIModel:
    """
    Devuelve la instancia única del modelo.
    Se crea la primera vez que se llama.
    """
    global _model_instance  # noqa: PLW0603
    if _model_instance is None:
        _model_instance = AIModel(model_path=settings.model_path)
    return _model_instance
