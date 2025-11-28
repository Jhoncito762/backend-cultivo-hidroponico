# test_tf.py
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Versión de TF:", tf.__version__)

# 1. Probar algo sencillo
m = tf.keras.applications.DenseNet121(weights=None)
print("DenseNet vacía creada OK")

# 2. Probar cargar tu modelo
modelo = load_model("DenseNet121_best_model.keras")
print("Modelo cargado OK")
