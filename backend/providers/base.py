from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseProvider(ABC):
    modal_type: str = "text"  # "text" | "image" | "audio" | "video"

    @abstractmethod
    async def generate(self, messages: list, params: dict) -> str:
        """非ストリーミング推論"""
        pass

    @abstractmethod
    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        """ストリーミング推論 (SSE)"""
        pass

    @abstractmethod
    def get_models(self) -> list[str]:
        """利用可能なモデル一覧"""
        pass

    @abstractmethod
    def validate_key(self) -> bool:
        """APIキーの有効性確認"""
        pass
