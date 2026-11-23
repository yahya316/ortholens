"""OrthoLens FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import pipeline
from .config import settings
from .model import Models, load_models
from .schemas import AnalysisResponse, HealthResponse

logging.basicConfig(level=logging.INFO)  # so DEBUG_QUALITY output is visible
logger = logging.getLogger("ortholens")


def missing_message(models: Models) -> str:
    """503 message naming every missing model file."""
    names = ", ".join(models.missing)
    return (f"Model file(s) missing or unusable: {names}. Place best.pt, resnet50.pt and combo_config.json "
            f"in {settings.MODELS_DIR} (see README) and restart the server.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load YOLO, ResNet50 and the combo config once at startup; start anyway if files are missing."""
    app.state.models = load_models(settings)
    yield


app = FastAPI(title="OrthoLens API", version="0.2.0", lifespan=lifespan,
              description="Screening and teaching aid for suspected wrist/forearm fractures. Not a diagnostic tool.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _models(request: Request) -> Models:
    """Return the app's models, or an empty set if startup has not run."""
    m = getattr(request.app.state, "models", None)
    if m is None:
        from .model import Detector, ResNetClassifier
        m = Models(Detector(settings), ResNetClassifier(settings), None,
                   settings.COMBO_CONFIG_PATH.name, settings.MODEL_PATH.name, settings.RESNET_PATH.name)
        request.app.state.models = m
    return m


def _error(status_code: int, message: str, models: Models | None = None) -> JSONResponse:
    """Return a JSON error using the standard envelope (notice included)."""
    return JSONResponse(status_code=status_code, content=pipeline.error_envelope(message, settings, models))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn FastAPI validation errors (e.g. missing 'file' field) into clean messages."""
    return _error(400, "No image was uploaded. Send the image in the multipart field named 'file'.")


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, Any]:
    """Report which model files are loaded and which device is used."""
    m = _models(request)
    return {"models_loaded": m.status, "device": m.device}


@app.post("/api/analyses", response_model=AnalysisResponse,
          responses={400: {"model": AnalysisResponse}, 413: {"model": AnalysisResponse},
                     415: {"model": AnalysisResponse}, 503: {"model": AnalysisResponse}})
def create_analysis(request: Request, file: UploadFile = File(...), mode: str | None = Form(None)) -> Any:
    """Analyse an uploaded photo of a wrist/forearm X-ray film.

    Optional form field ``mode``: ``screening`` (default from DECISION_MODE) or ``balanced``.
    The upload is processed in memory only; the original file name is never used.
    """
    m = _models(request)
    try:
        data = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
        pipeline.validate_and_decode(data, settings)  # fail fast with a precise 4xx
        pipeline.resolve_mode(mode, settings)
        if m.missing:
            return _error(503, missing_message(m), m)
        return pipeline.analyze(data, m, settings, mode)
    except pipeline.PipelineError as exc:
        return _error(exc.http_status, exc.message, m)
    except Exception:
        logger.exception("Analysis failed")
        return _error(500, "The analysis failed unexpectedly. Please try another image.", m)
    finally:
        file.file.close()
