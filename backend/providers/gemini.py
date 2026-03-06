import os
from typing import AsyncGenerator

from .base import BaseProvider

AVAILABLE_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]


class GeminiProvider(BaseProvider):
    modal_type: str = "text"

    def __init__(self):
        import google.generativeai as genai
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _build_contents(self, messages: list) -> tuple[str | None, list]:
        """Convert OpenAI-style messages to Gemini contents format."""
        system_prompt = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        return system_prompt, contents

    async def generate(self, messages: list, params: dict) -> str:
        import google.generativeai as genai
        model_name = params.get("model", "gemini-1.5-flash")
        system_prompt, contents = self._build_contents(messages)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        generation_config = genai.types.GenerationConfig(
            temperature=params.get("temperature", 0.7),
            max_output_tokens=params.get("max_tokens", 2048),
            top_p=params.get("top_p", 1.0),
        )
        response = model.generate_content(contents, generation_config=generation_config)
        return response.text

    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        import google.generativeai as genai
        model_name = params.get("model", "gemini-1.5-flash")
        system_prompt, contents = self._build_contents(messages)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        generation_config = genai.types.GenerationConfig(
            temperature=params.get("temperature", 0.7),
            max_output_tokens=params.get("max_tokens", 2048),
            top_p=params.get("top_p", 1.0),
        )
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def get_models(self) -> list[str]:
        return AVAILABLE_MODELS

    def validate_key(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            list(genai.list_models())
            return True
        except Exception:
            return False
