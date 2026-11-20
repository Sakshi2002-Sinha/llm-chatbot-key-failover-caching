"""Application settings, parsed once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _csv(name: str, default: str = "") -> list[str]:
    return [p.strip() for p in os.getenv(name, default).split(",") if p.strip()]


def sqlite_path_from_url(url: str) -> str:
    """Translate a SQLAlchemy-style SQLite URL into a filesystem path.

    sqlite:////data/app.db   -> /data/app.db   (absolute)
    sqlite:///./data/app.db  -> ./data/app.db  (relative)
    sqlite:///:memory:       -> :memory:
    A bare path is returned unchanged.
    """
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return url[len(prefix):] or ":memory:"
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):] or ":memory:"
    return url


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=list)
    model: str = "gemini/gemini-3.5-flash-lite"
    models: list[str] = field(default_factory=list)
    database_path: str = "./data/app.db"
    allowed_origins: list[str] = field(default_factory=list)
    enable_failure_simulation: bool = False
    default_system_prompt: str = "You are a helpful assistant."
    cache_ttl_seconds: int = 86400
    llm_timeout_seconds: int = 30
    key_cooldown_seconds: int = 30
    max_history_messages: int = 40
    max_output_tokens: int = 1024

    def model_for_key(self, index: int) -> str:
        """Model for the key at ``index`` (0-based). LLM_MODELS may hold one or N values."""
        if not self.models:
            return self.model
        if len(self.models) == 1:
            return self.models[0]
        return self.models[index] if index < len(self.models) else self.model


def load_settings() -> Settings:
    return Settings(
        api_keys=_csv("LLM_API_KEYS"),
        model=os.getenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite").strip(),
        models=_csv("LLM_MODELS"),
        database_path=sqlite_path_from_url(os.getenv("DATABASE_URL", "sqlite:///./data/app.db")),
        allowed_origins=_csv("ALLOWED_ORIGINS", "http://localhost:3000"),
        enable_failure_simulation=_bool("ENABLE_FAILURE_SIMULATION"),
        default_system_prompt=os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful assistant."),
        cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 86400),
        llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 30),
        key_cooldown_seconds=_int("KEY_COOLDOWN_SECONDS", 30),
        max_history_messages=_int("MAX_HISTORY_MESSAGES", 40),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS", 1024),
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()
