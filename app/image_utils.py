# app/image_utils.py
import base64
import io

from PIL import Image


class ImageDecodingError(Exception):
    """Error específico al decodificar la imagen base64."""


def decode_base64_image(image_b64: str) -> Image.Image:
    """
    Recibe una cadena base64 (con o sin prefijo data:image/...) y retorna un objeto PIL.Image.

    :param image_b64: Cadena base64 de la imagen.
    :return: Imagen PIL en formato RGB.
    :raises ImageDecodingError: Si no se puede decodificar.
    """
    try:
        # Si viene con prefijo "data:image/jpeg;base64,AAAA..."
        if "," in image_b64:
            _, image_b64 = image_b64.split(",", maxsplit=1)

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")  # normalizar
        return image
    except Exception as exc:  # noqa: BLE001
        raise ImageDecodingError(f"No se pudo decodificar la imagen: {exc}") from exc

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convierte una imagen PIL a base64 (sin prefijo data:image/...).

    :param image: Imagen PIL.
    :param format: Formato de salida, por defecto PNG.
    :return: Cadena base64.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

