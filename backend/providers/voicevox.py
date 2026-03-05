import os
import subprocess
import tempfile
from typing import AsyncGenerator

import aiohttp

from .base import BaseProvider


class VoiceVoxProvider(BaseProvider):
    modal_type: str = "audio"

    def __init__(self):
        self.base_url = os.getenv("VOICEVOX_URL", "http://localhost:50021")

    # ------------------------------------------------------------------
    # BaseProvider interface (text methods are no-ops for audio provider)
    # ------------------------------------------------------------------

    async def generate(self, messages: list, params: dict) -> str:
        """Synthesize speech and return path to the output MP3 file."""
        text = messages[-1].get("content", "") if messages else ""
        speaker = params.get("speaker_id", 1)
        speed = params.get("speed", 1.0)
        mp3_path = await self.synthesize_mp3(text, speaker_id=speaker, speed=speed)
        return mp3_path

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        """Not applicable for TTS — yields the final file path."""
        result = await self.generate(messages, params)
        yield result

    def get_models(self) -> list[str]:
        return ["voicevox"]

    def validate_key(self) -> bool:
        """VoiceVox uses no API key — check engine reachability instead."""
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/version", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # VoiceVox-specific methods
    # ------------------------------------------------------------------

    async def get_speakers(self) -> list[dict]:
        """Return speaker list from VoiceVox engine."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/speakers") as resp:
                resp.raise_for_status()
                return await resp.json()

    async def synthesize_wav(self, text: str, speaker_id: int = 1, speed: float = 1.0) -> bytes:
        """Synthesize speech and return raw WAV bytes."""
        async with aiohttp.ClientSession() as session:
            # Step 1: generate audio query
            params = {"text": text, "speaker": speaker_id}
            async with session.post(
                f"{self.base_url}/audio_query", params=params
            ) as resp:
                resp.raise_for_status()
                query = await resp.json()

            query["speedScale"] = speed

            # Step 2: synthesize
            async with session.post(
                f"{self.base_url}/synthesis",
                params={"speaker": speaker_id},
                json=query,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def synthesize_mp3(
        self, text: str, speaker_id: int = 1, speed: float = 1.0
    ) -> str:
        """Synthesize speech and convert to MP3. Returns path to temp file."""
        wav_bytes = await self.synthesize_wav(text, speaker_id=speaker_id, speed=speed)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file.write(wav_bytes)
            wav_path = wav_file.name

        mp3_path = wav_path.replace(".wav", ".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", mp3_path],
            check=True,
            capture_output=True,
        )
        os.unlink(wav_path)
        return mp3_path
