"""LiteLLM gateway: key pool, automatic failover and response caching.

LiteLLM provides the provider-agnostic ``acompletion`` call, the exception
mapping (RateLimitError, ServiceUnavailableError, ...), usage accounting and the
response cache. This module adds a small, deterministic failover loop on top so
that every reply can be attributed to an exact ``key_id`` and
``fallback_attempts`` count, and so that the X-Simulate-Failure hook can make a
specific key fail for one request only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from .config import Settings

logger = logging.getLogger("chatbot.llm")

litellm.suppress_debug_info = True
litellm.drop_params = True  # ignore params a provider does not support instead of failing

# Errors where trying another key cannot help: the *input* is the problem.
_NON_FAILOVER_ERRORS: tuple[type[BaseException], ...] = tuple(
    e
    for e in (
        getattr(litellm, "ContextWindowExceededError", None),
        getattr(litellm, "ContentPolicyViolationError", None),
    )
    if isinstance(e, type)
)


class SimulatedProviderError(Exception):
    """Raised in place of a provider call when X-Simulate-Failure targets a key."""

    def __init__(self, key_id: str, status_code: int) -> None:
        super().__init__(f"simulated HTTP {status_code} for {key_id}")
        self.key_id = key_id
        self.status_code = status_code


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        super().__init__("all providers unavailable")
        self.failures = failures


@dataclass
class LLMResult:
    reply: str
    model: str
    resolution_source: str  # LIVE_INFERENCE | LLM_CACHE
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


@dataclass
class ApiKey:
    key_id: str
    secret: str = field(repr=False)
    model: str
    cooldown_until: float = 0.0

    def is_cooling(self, now: float) -> bool:
        return self.cooldown_until > now


# --------------------------------------------------------------------------- cache


def cache_key_for(model: str, messages: list[dict[str, Any]]) -> str:
    """Cache key = model + the full message list sent to the LLM.

    Deliberately excludes the API key, so a reply cached through key_1 is
    reused when the same prompt is later served through key_2.
    """
    canonical = json.dumps(
        {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "chat:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _custom_get_cache_key(*args: Any, **kwargs: Any) -> str:
    return cache_key_for(kwargs.get("model", ""), kwargs.get("messages") or [])


def configure_cache(settings: Settings) -> bool:
    """Enable LiteLLM's built-in in-memory response cache. Returns True when enabled."""
    try:
        from litellm.caching.caching import Cache  # litellm >= 1.50
    except ImportError:  # pragma: no cover - older layout
        from litellm.caching import Cache  # type: ignore

    cache = Cache(type="local", ttl=settings.cache_ttl_seconds)
    cache.get_cache_key = _custom_get_cache_key  # type: ignore[method-assign]
    litellm.cache = cache
    return litellm.cache is not None


async def warm_up() -> None:
    """Exercise the live and cache-hit code paths once at startup (offline, mock provider).

    LiteLLM lazily imports a sizeable chunk of code the first time it serves a
    cache hit (~250 ms). Paying that at boot keeps the very first real cache hit
    fast, so cache hits are reliably faster than live calls.
    """
    messages = [{"role": "user", "content": "__warmup__"}]
    try:
        for _ in range(2):  # 1st: live path + cache write, 2nd: cache-hit path
            response = await litellm.acompletion(
                model="gpt-4o-mini", messages=messages, api_key="warmup", mock_response="ok", caching=True
            )
            if not _is_cache_hit(response):
                await _ensure_cached(response, "gpt-4o-mini", messages)
    except Exception:  # noqa: BLE001 - warm-up is best effort
        logger.warning("LiteLLM warm-up failed", exc_info=True)


# --------------------------------------------------------------------------- pool


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _is_throttle_or_outage(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None and (code == 429 or code >= 500):
        return True
    return isinstance(
        exc,
        tuple(
            e
            for e in (
                getattr(litellm, "RateLimitError", None),
                getattr(litellm, "ServiceUnavailableError", None),
                getattr(litellm, "InternalServerError", None),
                getattr(litellm, "APIConnectionError", None),
                getattr(litellm, "Timeout", None),
            )
            if isinstance(e, type)
        ),
    )


class KeyPool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys: list[ApiKey] = [
            ApiKey(key_id=f"key_{i + 1}", secret=secret, model=settings.model_for_key(i))
            for i, secret in enumerate(settings.api_keys)
        ]
        # Dev/test only: route through LiteLLM's mock provider instead of the network.
        self.mock_response: str | None = os.getenv("LLM_MOCK_RESPONSE") or None

    @property
    def key_ids(self) -> list[str]:
        return [k.key_id for k in self.keys]

    def __len__(self) -> int:
        return len(self.keys)

    def _ordered_keys(self) -> list[ApiKey]:
        """Configured order, but keys in cooldown go last (still tried as a last resort)."""
        now = time.monotonic()
        healthy = [k for k in self.keys if not k.is_cooling(now)]
        cooling = [k for k in self.keys if k.is_cooling(now)]
        return healthy + cooling

    async def _call(self, key: ApiKey, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": key.model,
            "messages": messages,
            "api_key": key.secret,
            "timeout": self.settings.llm_timeout_seconds,
            "num_retries": 0,
            "max_tokens": self.settings.max_output_tokens,
            "caching": True,
        }
        if self.mock_response is not None:
            kwargs["mock_response"] = self.mock_response
        return await litellm.acompletion(**kwargs)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        simulated_failures: dict[str, int] | None = None,
    ) -> LLMResult:
        simulated_failures = simulated_failures or {}
        failures: list[dict[str, Any]] = []

        for key in self._ordered_keys():
            try:
                if key.key_id in simulated_failures:
                    raise SimulatedProviderError(key.key_id, simulated_failures[key.key_id])
                response = await self._call(key, messages)
                reply = _extract_text(response)
                if not reply:
                    raise ValueError("provider returned an empty reply")
            except _NON_FAILOVER_ERRORS as exc:  # type: ignore[misc]
                logger.warning("%s rejected the request (%s); not failing over", key.key_id, type(exc).__name__)
                failures.append({"key_id": key.key_id, "error": type(exc).__name__, "status": _status_code(exc)})
                raise AllProvidersUnavailable(failures) from exc
            except Exception as exc:  # noqa: BLE001 - every key-level failure triggers failover
                status = _status_code(exc)
                failures.append({"key_id": key.key_id, "error": type(exc).__name__, "status": status})
                simulated = isinstance(exc, SimulatedProviderError)
                logger.warning(
                    "%s failed (%s%s)%s; trying next key",
                    key.key_id,
                    type(exc).__name__,
                    f" {status}" if status else "",
                    " [simulated]" if simulated else "",
                )
                if not simulated and _is_throttle_or_outage(exc):
                    key.cooldown_until = time.monotonic() + self.settings.key_cooldown_seconds
                continue

            cache_hit = _is_cache_hit(response)
            input_tokens, output_tokens = _extract_usage(response, key.model, messages, reply)
            if not cache_hit:
                key.cooldown_until = 0.0  # a success clears any earlier back-off
                await _ensure_cached(response, key.model, messages)
            return LLMResult(
                reply=reply,
                model=key.model,
                resolution_source="LLM_CACHE" if cache_hit else "LIVE_INFERENCE",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=None if cache_hit else key.key_id,
                fallback_attempts=0 if cache_hit else len(failures),
            )

        raise AllProvidersUnavailable(failures)


# --------------------------------------------------------------------------- helpers


async def _ensure_cached(response: Any, model: str, messages: list[dict[str, Any]]) -> None:
    """Write the fresh response to LiteLLM's cache before we return.

    LiteLLM writes the cache from a background task after the call completes, so
    an identical request arriving a few milliseconds later can still miss. We
    await the same write explicitly (same cache, same key function) to make the
    "second identical request is LLM_CACHE" guarantee deterministic. Re-writing
    an existing key is idempotent.
    """
    cache = litellm.cache
    if cache is None:
        return
    try:
        await cache.async_add_cache(response, model=model, messages=messages)
    except Exception:  # noqa: BLE001 - caching is best-effort, never fail the request
        logger.warning("explicit cache write failed", exc_info=True)



def _is_cache_hit(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None) or {}
    return bool(hidden.get("cache_hit"))


def _extract_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return (content or "").strip()


def _extract_usage(response: Any, model: str, messages: list[dict[str, Any]], reply: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion = getattr(usage, "completion_tokens", None) if usage is not None else None
    if prompt is None:
        try:
            prompt = litellm.token_counter(model=model, messages=messages)
        except Exception:  # noqa: BLE001
            prompt = 0
    if completion is None:
        try:
            completion = litellm.token_counter(model=model, text=reply)
        except Exception:  # noqa: BLE001
            completion = 0
    return int(prompt or 0), int(completion or 0)
