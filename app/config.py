# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    app_name: str = Field(default="AI Python Flask Backend")
    api_v1_prefix: str = Field(default="/api/v1")
    environment: Literal["dev", "prod", "test"] = Field(default="dev")
    log_level: str = Field(default="INFO")

    # Modelo DenseNet
    model_path: str | None = Field(
        default="DenseNet121_best_model.keras",
        description="Ruta al modelo de IA (.keras / .h5 / SavedModel)",
    )

    # 👉 NUEVO: modelo YOLO11 de segmentación
    yolo_seg_model_path: str | None = Field(
        default="best.pt",  # cámbialo por el nombre de tu archivo
        description="Ruta al modelo YOLO de segmentación (.pt)",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
