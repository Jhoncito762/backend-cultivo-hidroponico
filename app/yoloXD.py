from ultralytics import YOLO
import cv2
import os
from PIL import Image
from .image_utils import decode_base64_image, encode_image_to_base64
class YoloKhe():
    def run_inference(base64):
        # 1. Cargar el modelo entrenado
        # **IMPORTANTE:** Reemplaza esta ruta con la ubicación real de tu archivo 'best.pt'
        MODEL_PATH="/home/usco/Downloads/}/flask2-backend/best.pt"
        
        try:
            model = YOLO(MODEL_PATH) 
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el modelo en la ruta: {MODEL_PATH}")
            print("Asegúrate de que la ruta sea correcta y que el entrenamiento haya terminado.")
            return

        # 2. Definir la fuente de las imágenes para la inferencia
        # Puedes usar:
        # - Una ruta a un archivo de imagen: 'imagen_de_prueba.jpg'
        # - Una ruta a una carpeta: 'ruta/a/mi/carpeta_con_imagenes'
        # - Un flujo de video o cámara (ej: '0' para webcam)
        SOURCE_IMAGE = decode_base64_image(base64) # <--- CAMBIA ESTO

        # 3. Ejecutar la predicción
        # El método predict maneja la carga, el preprocesamiento, la ejecución y el post-procesamiento.
        print(f"🔍 Ejecutando inferencia en: {SOURCE_IMAGE}")
        results = model.predict(
            source=SOURCE_IMAGE,
            conf=0.25,     # Umbral de confianza mínimo (ajústalo)
            iou=0.7,       # Umbral de IOU para Non-Maximum Suppression (NMS)
            save=True,     # Guarda la imagen con las predicciones dibujadas
            project='YOLO11_Inferencia_Project', # Carpeta donde se guardan los resultados
            name='run_test',
            show=False     # No mostrar la ventana emergente (pon True si usas un entorno con GUI)
        )

        # 4. Procesar y/o mostrar los resultados (Opcional, pero útil)
        for r in results:
            # 'r' es un objeto 'Results' que contiene toda la información de la predicción de UNA imagen
            
            # Acceder a la imagen resultante (con máscaras, cajas y etiquetas dibujadas)
            # La imagen está en formato NumPy (RGB)
            im_array = r.plot()
            
            # Información de los objetos detectados:
            print(f"\n✅ Predicciones para una imagen:")
            print(f"   Clases detectadas: {r.names}")
            print(f"   Número de instancias detectadas: {len(r.boxes)}")
            
            # Ejemplo de cómo obtener las máscaras
            if r.masks is not None:
                # r.masks.data contiene las máscaras en formato tensor
                # r.masks.xy contiene los polígonos de las máscaras
                print(f"   Máscaras de segmentación encontradas. Forma del tensor de datos: {r.masks.data.shape}")
                
            # El archivo guardado estará en 'YOLO11_Inferencia_Project/run_test/'
            print(f"\n🖼️ Resultado visual guardado en: YOLO11_Inferencia_Project/run_test/")
            
            # Convertir la imagen NumPy a PIL Image y luego a base64
            pil_image = Image.fromarray(im_array)
            base64_result = encode_image_to_base64(pil_image, format="JPEG")
            
            return base64_result

