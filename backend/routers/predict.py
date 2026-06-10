"""REST prediction endpoints — raw sensors, features, metrics, reset."""

from fastapi import APIRouter, Request

from backend.models.schemas import (
    FeatureInput,
    PredictionResponse,
    RawSensorInput,
)

router = APIRouter(prefix="/api")


@router.post("/predict/raw", response_model=PredictionResponse)
async def predict_from_raw(data: RawSensorInput, request: Request):
    """Predict BSL sign from raw sensor values (manual form / ESP32 REST)."""
    service = request.app.state.prediction_service

    right_packet = {
        "flex": data.right_flex,
        "fsr": data.right_fsr,
        "contact_bitmask": data.right_bitmask,
        "quaternion": data.right_quaternion,
        "accel": data.right_accel,
    }
    left_packet = {
        "flex": data.left_flex,
        "fsr": data.left_fsr,
        "contact_bitmask": data.left_bitmask,
        "quaternion": data.left_quaternion,
        "accel": data.left_accel,
    }

    # One-shot path (bypasses debounce + motion routing). Streaming WS uses
    # service.predict_from_raw which keeps debounce/motion routing for stability.
    result = service.predict_oneshot_from_raw(right_packet, left_packet)

    # Trigger TTS for predicted sign
    tts = request.app.state.tts_service
    if result.get("sign"):
        result["audio_url"] = tts.speak(result["sign"])

    return result


@router.post("/predict/features", response_model=PredictionResponse)
async def predict_from_features(data: FeatureInput, request: Request):
    """Predict BSL sign from pre-computed NUM_FEATURES vector."""
    service = request.app.state.prediction_service
    result = service.predict_oneshot_from_features(data.features)

    # Trigger TTS for predicted sign
    tts = request.app.state.tts_service
    if result.get("sign"):
        result["audio_url"] = tts.speak(result["sign"])

    return result


@router.get("/metrics")
async def get_metrics(request: Request):
    """Get accumulated performance metrics."""
    return request.app.state.prediction_service.get_metrics()


@router.get("/predict/diagnostics")
async def get_prediction_diagnostics(request: Request):
    """Get prediction WebSocket packet diagnostics."""
    lan_ips = getattr(request.app.state, "lan_ips", [])
    return request.app.state.prediction_diagnostics.get_diagnostics(server_lan_ips=lan_ips)


@router.post("/reset")
async def reset_predictor(request: Request):
    """Reset predictor state and metrics."""
    request.app.state.prediction_service.reset()
    return {"status": "reset"}
