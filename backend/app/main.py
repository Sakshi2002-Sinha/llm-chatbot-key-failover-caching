"""FastAPI application: /health, /chat, /sessions/{session_id}/messages."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .db import Database
from .llm import AllProvidersUnavailable, KeyPool, configure_cache, warm_up
from .schemas import ChatMetadata, ChatRequest, ChatResponse, HealthResponse
from .simulation import SimulationHeaderError, parse_simulation_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db.init()
        app.state.cache_enabled = configure_cache(settings)
        await warm_up()
        logger.info(
            "started: keys=%d models=%s cache=%s simulation=%s db=%s",
            len(app.state.pool),
            sorted({k.model for k in app.state.pool.keys}),
            app.state.cache_enabled,
            settings.enable_failure_simulation,
            settings.database_path,
        )
        yield

    app = FastAPI(
        title="LLM Chatbot with Key Failover and Response Caching",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = Database(settings.database_path)
    app.state.pool = KeyPool(settings)
    app.state.cache_enabled = False
    session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Simulate-Failure"],
    )

    # ------------------------------------------------------------ error hygiene

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = []
        for err in exc.errors():
            err = dict(err)
            err.pop("ctx", None)  # may hold non-JSON-serialisable exception objects
            err.pop("input", None)  # never echo user payloads back
            err.pop("url", None)
            errors.append(err)
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled_error(_: Request, exc: Exception):
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    # ------------------------------------------------------------ routes

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "llm-chatbot-backend",
            "endpoints": ["GET /health", "POST /chat", "GET /sessions/{session_id}/messages"],
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        db_ok = await run_in_threadpool(app.state.db.ping)
        keys = len(app.state.pool)
        body = {"status": "ok", "keys_configured": keys, "cache_enabled": bool(app.state.cache_enabled)}
        if not db_ok or keys == 0:
            body["status"] = "degraded"
            return JSONResponse(status_code=503, content=body)
        return body

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    ):
        started = time.perf_counter()
        pool: KeyPool = app.state.pool
        db: Database = app.state.db

        simulated: dict[str, int] = {}
        if settings.enable_failure_simulation and x_simulate_failure:
            try:
                simulated = parse_simulation_header(x_simulate_failure, pool.key_ids)
            except SimulationHeaderError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None

        async with session_locks[payload.session_id]:
            session = await run_in_threadpool(db.get_session, payload.session_id)
            history = await run_in_threadpool(db.get_messages, payload.session_id) if session else []

            system_prompt = (
                payload.system_prompt
                or (session or {}).get("system_prompt")
                or settings.default_system_prompt
            )
            window = history[-settings.max_history_messages :] if settings.max_history_messages > 0 else []
            messages = [{"role": "system", "content": system_prompt}]
            messages += [{"role": m["role"], "content": m["content"]} for m in window]
            messages.append({"role": "user", "content": payload.message})

            try:
                result = await pool.complete(messages, simulated)
            except AllProvidersUnavailable as exc:
                logger.error("all providers unavailable for session %s: %s", payload.session_id, exc.failures)
                raise HTTPException(status_code=503, detail="all providers unavailable") from None

            latency_ms = int(round((time.perf_counter() - started) * 1000))
            metadata = ChatMetadata(
                resolution_source=result.resolution_source,
                model=result.model,
                latency_ms=latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                key_id=result.key_id,
                fallback_attempts=result.fallback_attempts,
            )
            await run_in_threadpool(
                db.save_turn,
                payload.session_id,
                payload.system_prompt,
                payload.message,
                result.reply,
                metadata.model_dump(),
            )

        return ChatResponse(session_id=payload.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str):
        db: Database = app.state.db
        session = await run_in_threadpool(db.get_session, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        messages = await run_in_threadpool(db.get_messages, session_id)
        return {"session_id": session_id, "messages": messages}

    return app


app = create_app()
