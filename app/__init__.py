# app/__init__.py
from flask import Flask

from .config import settings
from .logging_config import setup_logging
from .routes import api_bp


def create_app() -> Flask:
    """
    Crea y configura la instancia de Flask.
    """
    setup_logging()

    app = Flask(settings.app_name)
    app.config["ENVIRONMENT"] = settings.environment
    app.config["APP_NAME"] = settings.app_name

    # Registrar Blueprint bajo el prefijo de la API
    app.register_blueprint(api_bp, url_prefix=settings.api_v1_prefix)

    return app
