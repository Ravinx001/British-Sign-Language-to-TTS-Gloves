"""Pydantic request/response models for the BSL prediction API."""

from typing import Optional

from pydantic import BaseModel, Field


class RawSensorInput(BaseModel):
    """Raw sensor data from both gloves (same format as ESP32 JSON packets)."""

    right_flex: list[int] = Field(
        ..., min_length=5, max_length=5,
        description="Right flex ADC values [thumb..pinky] (800–3800)",
    )
    right_fsr: list[int] = Field(
        ..., min_length=3, max_length=3,
        description="Right FSR ADC values [thumb, index, pad] (0–4095)",
    )
    right_bitmask: int = Field(
        default=0, ge=0, le=31,
        description="Right touch bitmask (0–31). Optional — capacitive touch hardware removed; field kept for schema compat.",
    )
    right_quaternion: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Right orientation [w, x, y, z]",
    )
    right_accel: list[float] = Field(
        default=[0.0, 0.0, 9.8], min_length=3, max_length=3,
        description="Right accelerometer [x, y, z] m/s²",
    )

    left_flex: list[int] = Field(
        ..., min_length=5, max_length=5,
        description="Left flex ADC values [thumb..pinky] (800–3800)",
    )
    left_fsr: list[int] = Field(
        ..., min_length=3, max_length=3,
        description="Left FSR ADC values [thumb, index, pad] (0–4095)",
    )
    left_bitmask: int = Field(
        default=0, ge=0, le=31,
        description="Left touch bitmask (0–31). Optional — capacitive touch hardware removed; field kept for schema compat.",
    )
    left_quaternion: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Left orientation [w, x, y, z]",
    )
    left_accel: list[float] = Field(
        default=[0.0, 0.0, 9.8], min_length=3, max_length=3,
        description="Left accelerometer [x, y, z] m/s²",
    )


class FeatureInput(BaseModel):
    """Pre-computed NUM_FEATURES vector input (26 — see ml/config.py FEATURE_NAMES)."""

    features: list[float] = Field(
        ..., min_length=26, max_length=26,
        description="26 ML features (see ml/config.py FEATURE_NAMES)",
    )
    accel: list[float] = Field(
        default=[0.0, 0.0, 9.8], min_length=3, max_length=3,
        description="Accelerometer [x, y, z] m/s² for motion routing",
    )


class PredictionResponse(BaseModel):
    """Prediction result returned by all prediction endpoints."""

    sign: Optional[str] = None
    display_label: Optional[str] = None
    is_rest: bool = False
    rest_pose: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: float
    route: Optional[str] = None
    audio_url: Optional[str] = None
    # Diagnostic fields populated by the streaming + one-shot paths. Optional
    # so legacy callers that don't supply them still validate.
    top_k: Optional[list[dict]] = None
    calibration: Optional[dict] = None


class SettingsUpdate(BaseModel):
    """Partial settings update (all fields optional)."""

    static_confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
    )
    dynamic_confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
    )
    motion_energy_threshold: Optional[float] = Field(
        default=None, ge=0.0,
    )
    tts_enabled: Optional[bool] = None
    tts_rate: Optional[int] = Field(default=None, ge=50, le=400)
    tts_volume: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SettingsResponse(BaseModel):
    """Current settings."""

    static_confidence_threshold: float
    dynamic_confidence_threshold: float
    motion_energy_threshold: float
    tts_enabled: bool
    tts_engine: str
    tts_rate: int
    tts_volume: float


class ConnectionInfo(BaseModel):
    """Single ESP32 client connection info."""

    client_id: str
    endpoint: str = "predict"
    hand_id: int
    hand: str
    ip: str
    status: str
    packet_count: int
    avg_latency_ms: float
    battery_pct: Optional[float] = None
    connected_at: float
    last_seen: float


class ConnectionsResponse(BaseModel):
    """All active ESP32 connections."""

    connections: list[ConnectionInfo]
    total: int
