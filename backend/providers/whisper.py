"""Whisper STT provider.

Wraps the OpenAI Whisper API for speech-to-text transcription.
This is a standalone utility class — it does not extend BaseProvider
because STT is not a chat-style inference task.
"""
from __future__ import annotations

import os
from pathlib import Path


class WhisperProvider:
    """OpenAI Whisper API wrapper for speech-to-text."""

    MODEL = "whisper-1"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def _get_client(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set — required for Whisper STT")
        from openai import OpenAI
        return OpenAI(api_key=self.api_key)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text.

        Parameters
        ----------
        audio_path:
            Path to the audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm).
        language:
            Optional ISO-639-1 language code (e.g. "ja", "en").  If omitted,
            Whisper auto-detects the language.

        Returns
        -------
        Transcribed text string.
        """
        client = self._get_client()
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(path, "rb") as audio_file:
            kwargs: dict = {"model": self.MODEL, "file": audio_file}
            if language:
                kwargs["language"] = language
            response = client.audio.transcriptions.create(**kwargs)

        return response.text

    def is_available(self) -> bool:
        return bool(self.api_key)
