"""BSL Sign Language Gloves — FastAPI Backend."""

import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable for the ml package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.routers import audio, batch, calibration, collection, predict, settings, websocket
from backend.services.collection_service import DATA_DIR, CollectionService
from backend.services.connection_manager import ConnectionManager
from backend.services.prediction_diagnostics import PredictionDiagnostics
from backend.services.predictor import PredictionService
from backend.services.settings_store import SettingsStore
from backend.services.tts_service import TTSService


def _get_lan_ips() -> list[str]:
    """Return non-loopback IPv4 addresses on this machine."""
    try:
        _, _, addrs = socket.gethostbyname_ex(socket.gethostname())
        return [a for a in addrs if not a.startswith("127.")]
    except Exception:
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML models and services on startup."""
    print("Loading ML models...")
    app.state.prediction_service = PredictionService()
    app.state.connection_manager = ConnectionManager()
    app.state.tts_service = TTSService()
    app.state.collection_service = CollectionService()
    app.state.prediction_diagnostics = PredictionDiagnostics()
    app.state.settings_store = SettingsStore()
    app.state.lan_ips = _get_lan_ips()
    saved_settings = app.state.settings_store.load()
    if saved_settings:
        _apply_saved_settings(app, saved_settings)
    n_users = len(app.state.collection_service.get_users())
    print(f"Collection data: {DATA_DIR} ({n_users} user(s) in users.json)")
    if app.state.lan_ips:
        print(f"LAN IPs for ESP32: {', '.join(app.state.lan_ips)}")
    print(
        "For ESP32 gloves on Wi-Fi: start uvicorn with --host 0.0.0.0 (not 127.0.0.1) "
        "so devices can reach ws://<this-PC-LAN-IP>:8000/ws/predict or /ws/collect"
    )
    print(f"TTS engine: {app.state.tts_service.engine_name}")
    print("Models loaded. Server ready.")
    yield
    print("Shutting down...")


def _apply_saved_settings(app: FastAPI, settings: dict) -> None:
    """Apply persisted dashboard settings to runtime services."""
    service = app.state.prediction_service
    tts = app.state.tts_service
    if "static_confidence_threshold" in settings:
        service.static_confidence_threshold = settings["static_confidence_threshold"]
    if "dynamic_confidence_threshold" in settings:
        service.dynamic_confidence_threshold = settings["dynamic_confidence_threshold"]
    if "motion_energy_threshold" in settings:
        service.motion_energy_threshold = settings["motion_energy_threshold"]
    if "tts_enabled" in settings:
        tts.set_enabled(settings["tts_enabled"])
    if "tts_rate" in settings:
        tts.set_rate(settings["tts_rate"])
    if "tts_volume" in settings:
        tts.set_volume(settings["tts_volume"])


app = FastAPI(
    title="BSL Sign Language Gloves — Prediction API",
    description="Real-time BSL fingerspelling prediction from sensor glove data",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow browser API calls when page origin differs slightly (127.0.0.1 vs localhost, or alternate ports).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(websocket.router)
app.include_router(audio.router)
app.include_router(settings.router)
app.include_router(collection.router)
app.include_router(calibration.router)

# Serve static files (dashboard, audio)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
(static_dir / "audio").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def dashboard():
    """Serve the prediction dashboard SPA."""
    from fastapi.responses import FileResponse
    return FileResponse(str(static_dir / "index.html"))


@app.get("/collection")
async def collection_page():
    """Serve the data collection SPA."""
    from fastapi.responses import FileResponse

    return FileResponse(
        str(static_dir / "collection.html"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )
