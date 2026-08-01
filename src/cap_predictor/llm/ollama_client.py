"""Local Ollama HTTP client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(self.model in name or name.startswith(self.model) for name in models) or bool(
                    models
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama not available: %s", exc)
            return False

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if system:
            payload["system"] = system
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
                return (data.get("response") or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Ollama generate failed: {exc}") from exc
