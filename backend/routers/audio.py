"""Audio serving endpoint — returns the latest TTS-generated audio file."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api")


@router.get("/audio/latest")
async def get_latest_audio():
    """Return the most recently generated TTS audio file."""
    audio_dir = Path(__file__).parent.parent / "static" / "audio"
    for ext in ("wav", "mp3"):
        path = audio_dir / f"latest.{ext}"
        if path.is_file():
            media = "audio/wav" if ext == "wav" else "audio/mpeg"
            return FileResponse(str(path), media_type=media, filename=f"latest.{ext}")
    raise HTTPException(status_code=404, detail="No audio file available")
