"""Speech-to-Text API endpoint.

POST /v1/audio/transcriptions
  - multipart/form-data
  - Fields:
      file     : audio file (required)
      language : ISO-639-1 language code, e.g. "ja" (optional)
      provider : STT provider name, default "whisper" (optional)
  - Response: {"text": "..."}
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

router = APIRouter()

# Supported audio MIME types / extensions
_ALLOWED_SUFFIXES = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    provider: str = Form(default="whisper"),
) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. "
                   f"Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
        )

    if provider != "whisper":
        raise HTTPException(status_code=400, detail=f"Unknown STT provider: {provider}")

    from backend.providers.whisper import WhisperProvider
    stt = WhisperProvider()
    if not stt.is_available():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured — Whisper STT is unavailable",
        )

    # Save upload to a temp file, then transcribe
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = stt.transcribe(tmp_path, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"text": text}
