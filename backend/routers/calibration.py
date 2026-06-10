"""Per-user calibration endpoints.

Lets the dashboard capture per-user offsets (neutral pose) and per-flex
amplitude scales (optional reference letters A, B, C) so each wearer's signal
is mapped into the small 2-user training distribution before the StandardScaler
runs. Profiles persist under data/calibrations/<label>.json so returning users
can reload without re-capturing.

Flow:
    1) POST /api/calibrate/start  { "n_frames": 150 }
       → User holds both hands flat on the table, palms down, fingers straight
         for ~3 s. Frames stream in via WebSocket; PredictionService buffers
         them and finalises the offset against the training neutral baseline.
    2) GET  /api/calibrate/status
       → Poll until offset_applied=true.
    3) (Optional) Per reference letter A, B, C:
       POST /api/calibrate/reference/start  { "letter": "A" }
       (user signs A for ~2 s while frames stream in)
       POST /api/calibrate/reference/finish    # idempotent flush
    4) POST /api/calibrate/save  { "label": "<name>" }    # persist
       GET  /api/calibrate/profiles                       # enumerate saved
       POST /api/calibrate/load  { "label": "<name>" }    # reload
    5) POST /api/calibrate/clear  to discard.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/calibrate")


class CalibrationStartRequest(BaseModel):
    n_frames: int | None = Field(
        default=None, ge=10, le=1000,
        description=(
            "Number of consecutive frames to average for the neutral offset. "
            "Defaults to CALIBRATION['neutral_frames'] (150 = 3 s)."
        ),
    )


class CalibrationStatus(BaseModel):
    active: bool
    collected: int
    target: int
    state: str | None = None
    offset_applied: bool
    label: str | None = None
    refs_done: list[str] = []


class ReferenceStartRequest(BaseModel):
    letter: str = Field(
        ..., description="Reference letter to capture (e.g., 'A', 'B', 'C')."
    )


class SaveProfileRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)


class LoadProfileRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)


class ProfileSummary(BaseModel):
    label: str
    refs_done: list[str] = []
    captured_at: float | None = None


# ---------------------------------------------------------------------------
# Capture endpoints
# ---------------------------------------------------------------------------

@router.post("/start", response_model=CalibrationStatus)
async def start_calibration(req: CalibrationStartRequest, request: Request):
    service = request.app.state.prediction_service
    return service.start_calibration(n_frames=req.n_frames)


@router.get("/status", response_model=CalibrationStatus)
async def calibration_status(request: Request):
    service = request.app.state.prediction_service
    return service.get_calibration_status()


@router.post("/clear", response_model=CalibrationStatus)
async def clear_calibration(request: Request):
    service = request.app.state.prediction_service
    return service.clear_calibration()


@router.post("/reset")
async def reset_predictor(request: Request):
    """Clear debounce/motion state without dropping the calibration."""
    service = request.app.state.prediction_service
    return service.reset_state()


# ---------------------------------------------------------------------------
# Reference-letter refinement (optional)
# ---------------------------------------------------------------------------

@router.post("/reference/start", response_model=CalibrationStatus)
async def start_reference(req: ReferenceStartRequest, request: Request):
    service = request.app.state.prediction_service
    try:
        return service.start_reference_capture(req.letter)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reference/finish", response_model=CalibrationStatus)
async def finish_reference(request: Request):
    """Flush the current reference capture even if the target frame count was
    not yet reached. Idempotent if no capture is active."""
    service = request.app.state.prediction_service
    if service._calibrating and service._calib_buffer:
        service._finalize_calibration_step()
    return service.get_calibration_status()


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_profile(req: SaveProfileRequest, request: Request):
    service = request.app.state.prediction_service
    try:
        return service.save_calibration(req.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/profiles", response_model=list[ProfileSummary])
async def list_profiles(request: Request):
    service = request.app.state.prediction_service
    return service.list_calibration_profiles()


@router.post("/load")
async def load_profile(req: LoadProfileRequest, request: Request):
    service = request.app.state.prediction_service
    try:
        return service.load_calibration(req.label)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
