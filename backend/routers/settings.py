"""Settings & connection management endpoints."""

from fastapi import APIRouter, Request

from backend.models.schemas import (
    ConnectionsResponse,
    SettingsResponse,
    SettingsUpdate,
)

router = APIRouter(prefix="/api")


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(request: Request):
    """Get current runtime settings (thresholds, TTS config)."""
    service = request.app.state.prediction_service
    tts = request.app.state.tts_service

    thresholds = service.get_thresholds()
    tts_settings = tts.get_settings()

    return {**thresholds, **tts_settings}


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(update: SettingsUpdate, request: Request):
    """Update runtime settings. Only provided fields are changed."""
    service = request.app.state.prediction_service
    tts = request.app.state.tts_service
    cm = request.app.state.connection_manager

    if update.static_confidence_threshold is not None:
        service.static_confidence_threshold = update.static_confidence_threshold
    if update.dynamic_confidence_threshold is not None:
        service.dynamic_confidence_threshold = update.dynamic_confidence_threshold
    if update.motion_energy_threshold is not None:
        service.motion_energy_threshold = update.motion_energy_threshold
    if update.tts_enabled is not None:
        tts.set_enabled(update.tts_enabled)
    if update.tts_rate is not None:
        tts.set_rate(update.tts_rate)
    if update.tts_volume is not None:
        tts.set_volume(update.tts_volume)

    # Build current state and broadcast to dashboards
    thresholds = service.get_thresholds()
    tts_settings = tts.get_settings()
    current = {**thresholds, **tts_settings}
    request.app.state.settings_store.save(current)

    await cm.broadcast_to_dashboards({
        "type": "settings_changed",
        "settings": current,
    })

    return current


@router.get("/connections", response_model=ConnectionsResponse)
async def get_connections(request: Request):
    """List all active ESP32 glove connections."""
    cm = request.app.state.connection_manager
    conns = cm.get_all_connections()
    return {"connections": conns, "total": len(conns)}


@router.post("/connections/{client_id}/disconnect")
async def disconnect_client(client_id: str, request: Request):
    """Force-disconnect an ESP32 glove by client_id."""
    cm = request.app.state.connection_manager
    client = cm.esp32_clients.get(client_id)
    if not client:
        return {"status": "not_found", "client_id": client_id}
    try:
        await client.websocket.close()
    except Exception:
        pass
    await cm.disconnect_esp32(client_id)
    return {"status": "disconnected", "client_id": client_id}
