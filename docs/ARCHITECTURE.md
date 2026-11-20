# Architecture & Engineering Design

**Project:** LLM Chatbot with Key Failover and Response Caching
**Stack:** React (Vite) · Python 3.11 / FastAPI · LiteLLM · SQLite · Docker
**Status:** Implemented — this document is the design the code follows.

---

## 1. Goals and non-goals

### Goals
| # | Goal | How it is measured |
|---|------|--------------------|
| G1 | Multi-turn chat whose context lives on the server | Bot remembers earlier turns; `GET /sessions/{id}/messages` returns history from SQLite |
| G2 | Survive key failures without the user noticing | 429 / quota / 5xx on one key → next key answers, `fallback_attempts` > 0 |
| G3 | Clean failure when *every* key fails | `503 {"detail": "all providers unavailable"}`, nothing written to history |
| G4 | Cheaper and faster repeated prompts | Same model + same message list → `LLM_CACHE`, identical text, lower latency |
| G5 | Honest per-request telemetry | `latency_ms`, `input_tokens`, `output_tokens`, `key_id`, `model` on every reply |
| G6 | Ship it | `make up` locally; public frontend + backend URLs |

### Non-goals
- User accounts / auth (sessions are anonymous, client-generated ids).
- Streaming responses (would complicate the per-request metadata contract).
- Horizontal scaling of the backend (SQLite + in-process cache imply a single instance; see §9).

---

## 2. System context

```
┌──────────────────────┐      HTTPS / JSON       ┌───────────────────────────────┐      HTTPS      ┌───────────────────┐
│  Browser             │ ─────────────────────▶  │  FastAPI backend (uvicorn)    │ ─────────────▶ │  LLM provider     │
│  React SPA (Vite)    │  POST /chat             │                               │  via LiteLLM    │  (Gemini / Groq / │
│  served by nginx     │  GET  /sessions/{id}/…  │  ┌─────────┐  ┌────────────┐  │                 │   OpenAI …)       │
│  (port 3000)         │  GET  /health           │  │ KeyPool │  │ LiteLLM    │  │                 │  key_1, key_2, …  │
└──────────────────────┘ ◀─────────────────────  │  │ failover│─▶│ acompletion│  │ ◀───────────── └───────────────────┘
                              reply + metadata   │  └─────────┘  │ + Cache    │  │
                                                 │               └────────────┘  │
                                                 │  ┌─────────────────────────┐  │
                                                 │  │ SQLite  (/data/app.db)  │◀─┼── Docker named volume `chat-data`
                                                 │  └─────────────────────────┘  │
                                                 └───────────────────────────────┘
```

Two deployable units:

| Unit | Runtime image | Port | Hosting target |
|------|---------------|------|----------------|
| `frontend/` | `nginx:alpine` serving the Vite build | 3000 | Vercel / Netlify (static) |
| `backend/`  | `python:3.11-slim` + pre-built venv | 8000 | Render (Docker web service) |

---

## 3. Backend design

### 3.1 Module layout

```
backend/
├── app/
│   ├── main.py         # FastAPI app, routes, CORS, exception handlers, lifespan
│   ├── config.py       # Settings parsed from environment variables (single source of truth)
│   ├── schemas.py      # Pydantic request/response models = the API contract
│   ├── db.py           # SQLite access: schema bootstrap, sessions, messages
│   ├── llm.py          # KeyPool, failover loop, LiteLLM call + cache detection
│   └── simulation.py   # X-Simulate-Failure header parsing
└── tests/              # pytest suite, runs with no real API keys
```

Each module has a single responsibility; `main.py` only orchestrates.

### 3.2 Request lifecycle — `POST /chat`

```
 1. Pydantic validation ──────────────── invalid → 422 (FastAPI default shape)
 2. Parse X-Simulate-Failure ─────────── only if ENABLE_FAILURE_SIMULATION=true; malformed → 422
 3. t0 = perf_counter()
 4. acquire per-session asyncio.Lock      (serialises concurrent turns of one session)
 5. load session (system prompt) + history from SQLite
 6. effective system prompt = request.system_prompt ?? stored prompt ?? DEFAULT_SYSTEM_PROMPT
 7. messages = [system] + history[-MAX_HISTORY_MESSAGES:] + [user]
 8. KeyPool.complete(messages, simulated_failures)
        for each key in failover order:
            if key simulated to fail → raise SimulatedProviderError(status)   (counted as failure)
            try litellm.acompletion(model, messages, api_key=key, num_retries=0, timeout)
                 └─ LiteLLM cache checked first → hit returns stored response
            on failover-worthy error → record, maybe cool down key, continue
        all failed → AllProvidersUnavailable
 9. latency_ms = round((perf_counter() - t0) * 1000)
10. ONE transaction: upsert session, insert user msg, insert assistant msg + metadata JSON
11. return 200 {session_id, reply, metadata}

    AllProvidersUnavailable → 503 {"detail": "all providers unavailable"}  (step 10 never runs)
```

Key invariant: **history is only written after a successful reply**, in a single
transaction. A 422 or 503 therefore never leaves a dangling user message.

### 3.3 Key pool & failover

- `LLM_API_KEYS="k1,k2,…"` → keys get ids `key_1`, `key_2`, … in order. The raw key never
  leaves `KeyPool`; logs and responses only see the id.
- Optional `LLM_MODELS` lets each key target a different model (one value = same model for
  all keys). This allows "2 keys across one or more models".
- **Failover-worthy errors** (move on to the next key): every key-level failure, notably
  - HTTP 429 / `litellm.RateLimitError` (includes provider quota exhaustion)
  - HTTP 5xx / `InternalServerError`, `ServiceUnavailableError`, `APIConnectionError`, `Timeout`
  - auth errors (a revoked or invalid key is a key failure, not a user error; some providers
    such as Gemini report an invalid key as HTTP 400, so we don't treat 400 as "stop")
  - an empty reply from the provider
  - simulated failures from the test hook
- **Non-failover errors**: `ContextWindowExceededError` and `ContentPolicyViolationError`.
  Here the *input* is the problem and would fail on every key, so we stop early and return
  503 rather than burn the pool.
- **Cooldown (bonus):** a *real* 429/5xx puts that key in cooldown for `KEY_COOLDOWN_SECONDS`.
  Cooling keys are moved to the back of the order (still tried as a last resort), so the
  next request does not waste a round-trip on a key we know is throttled. Simulated
  failures never trigger cooldown — they are scoped to one request by contract.
- LiteLLM's own retries are disabled (`num_retries=0`) so failover is fast, deterministic,
  and `fallback_attempts` is exact.

**Why a thin loop on top of LiteLLM rather than `litellm.Router`?** The contract requires
exact attribution (`key_id`, `fallback_attempts`) and a per-request, per-key failure
injector. The Router's shuffle/cooldown/retry logic is opaque for that. We still use LiteLLM
for everything that is provider-specific: the unified `acompletion` API, the
exception-type mapping (`RateLimitError`, `ServiceUnavailableError`…), usage accounting,
and the response cache.

### 3.4 Response caching

- `litellm.cache = Cache(type="local")` — LiteLLM's built-in in-memory cache. (LiteLLM
  1.110 no longer ships the `disk` backend, and Redis is ruled out by the "no external
  services" rule, so in-memory is the right choice; it is per-process and resets on restart.)
- **Deterministic write:** LiteLLM writes the cache from a background task after the call,
  so an identical request arriving milliseconds later could still miss. After a live
  response we `await litellm.cache.async_add_cache(...)` with the same key function, which
  makes "the second identical request is `LLM_CACHE`" hold even for back-to-back calls.
- **Cache key = model + full message list** (system + history + new message). We install a
  custom key function that hashes exactly `{model, messages}` (SHA-256 of canonical JSON).
  This deliberately excludes `api_key`, so a reply cached via `key_1` is reused when the
  request is served via `key_2`.
- **Hit detection:** LiteLLM marks served-from-cache responses with
  `response._hidden_params["cache_hit"] = True` → `resolution_source = LLM_CACHE`,
  `key_id = null`, `fallback_attempts = 0`. Tokens are the usage stored with the cached
  response.
- A different message ⇒ different message list ⇒ different hash ⇒ never served from cache.
- TTL: `CACHE_TTL_SECONDS` (default 24 h).

### 3.5 Token accounting

- `input_tokens = usage.prompt_tokens`, `output_tokens = usage.completion_tokens` from the
  provider response of **this** call. Never summed across the conversation.
- If a provider omits usage, fall back to `litellm.token_counter` so the fields are always
  integers.

### 3.6 Latency

`latency_ms` is server-measured wall time for the whole request (history load → LLM/cache
→ persist), as an integer. A cache hit skips the network round-trip, so it is reliably
faster than a live call. Because LiteLLM lazily imports a large module graph on its *first*
cache hit (~250 ms), startup runs an offline warm-up (mock provider: one live call + one
cache hit) so the first real hit is also only a few ms.

---

## 4. Data model (SQLite)

```sql
CREATE TABLE IF NOT EXISTS sessions (
  session_id    TEXT PRIMARY KEY,
  system_prompt TEXT,                 -- per-session override, NULL = default
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content    TEXT NOT NULL,
  metadata   TEXT,                    -- JSON, assistant messages only
  created_at TEXT NOT NULL            -- server time, ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
```

- Schema is created on startup (`lifespan`), idempotently.
- Location from `DATABASE_URL` (`sqlite:////data/app.db` → `/data/app.db`); the directory is
  created if missing. In Docker, `/data` is the named volume `chat-data`.
- `PRAGMA journal_mode=WAL` for concurrent reads during writes; one short-lived connection
  per operation (SQLite connections are cheap and this avoids cross-thread sharing).
- The system prompt is stored on `sessions`, **not** as a message, so the history endpoint
  naturally excludes it.
- A session "exists" iff it has a `sessions` row, which is only created by a successful chat
  → unknown or failed-only sessions return 404.

---

## 5. API contract (summary)

| Method | Path | Success | Errors |
|--------|------|---------|--------|
| GET | `/health` | `{"status":"ok","keys_configured":N,"cache_enabled":true}` | 503 if DB unreachable or no keys |
| POST | `/chat` | `{session_id, reply, metadata{resolution_source, model, latency_ms, input_tokens, output_tokens, key_id, fallback_attempts}}` | 422 validation, 503 all providers unavailable |
| GET | `/sessions/{id}/messages` | `{session_id, messages:[{role, content, metadata?}]}` oldest first | 404 `session not found` |

Validation: `session_id` 1–128 chars; `message` 1–4000 chars after trim; optional
`system_prompt` (≤ 4000 chars).

Error hygiene: global handlers make **every** error JSON with a `detail` field; an
unexpected exception becomes `500 {"detail":"internal server error"}` and is logged
server-side — never a stack trace in the response.

### Failover test hook

`X-Simulate-Failure: key_1=429,key_2=500` or `*=503`. Statuses allowed: 429, 500, 502, 503.
Honoured only when `ENABLE_FAILURE_SIMULATION=true`. The listed keys raise a simulated
provider error *before* any network/cache call, for that request only. Unlisted keys behave
normally and return a real answer.

---

## 6. Frontend design

- Vite + React 18, no UI framework (small bundle, full control of `data-testid`s).
- `sessionId` generated with `crypto.randomUUID()` on every page load → new session per load.
- State: `messages[]` (user / assistant / error items), `pending`, `systemPrompt`.
- Each assistant bubble (`data-testid="assistant-message"`) renders a metadata strip:
  `meta-source`, `meta-model`, `meta-latency` ("812 ms"), `meta-input-tokens`,
  `meta-output-tokens`, plus key id and fallback count.
- Errors (422 / 503 / network) are rendered as a red system bubble in the chat — never thrown.
- Optional system prompt (`system-prompt-input`) is sent with each request.
- Bonus: a "Simulate failure" selector that sets `X-Simulate-Failure`, so failover can be
  demoed from the UI; and "Load history" which calls the history endpoint.
- Backend URL is `VITE_API_BASE_URL`, baked in at build time.

---

## 7. Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_API_KEYS` | — (required, ≥2) | Comma-separated key pool → `key_1`, `key_2`, … |
| `LLM_MODEL` | `gemini/gemini-3.5-flash-lite` | LiteLLM model name |
| `LLM_MODELS` | — | Optional per-key models, comma-separated, aligned with keys |
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLite location |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins, comma-separated |
| `ENABLE_FAILURE_SIMULATION` | `false` | Enables `X-Simulate-Failure` |
| `DEFAULT_SYSTEM_PROMPT` | "You are a helpful assistant." | Used when the session has none |
| `CACHE_TTL_SECONDS` | `86400` | Cache entry lifetime |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-key request timeout |
| `KEY_COOLDOWN_SECONDS` | `30` | Back-off for a key after a real 429/5xx |
| `MAX_HISTORY_MESSAGES` | `40` | History window sent to the model |
| `MAX_OUTPUT_TOKENS` | `1024` | Upper bound on reply length |
| `LLM_MOCK_RESPONSE` | — | Dev/test only: LiteLLM mock provider, no network |
| `VITE_API_BASE_URL` (frontend) | `http://localhost:8000` | Backend base URL |

---

## 8. Deployment

```
GitHub repo ──▶ Render (Docker, backend/Dockerfile)   env: LLM_API_KEYS, ALLOWED_ORIGINS=<vercel url>,
           │                                            ENABLE_FAILURE_SIMULATION=true, DATABASE_URL
           └──▶ Vercel (root: frontend/, Vite preset)  env: VITE_API_BASE_URL=<render url>
```

- Docker images are multi-stage: backend builds a venv in a `builder` stage and copies only
  the venv + app into a slim runtime (non-root user); frontend builds with Node and ships
  only static files on `nginx:alpine`.
- `docker-compose.yml` runs both with a named volume `chat-data` at `/data`;
  `make up` / `make down` wrap it.

---

## 9. Trade-offs and risks

| Decision | Trade-off | Mitigation / future work |
|----------|-----------|--------------------------|
| SQLite | Single writer, single instance | WAL mode, per-session locks; move to Postgres if scaling out |
| In-memory cache | Lost on restart, per-process | Fine for one instance; a shared cache (e.g. Redis) would be the next step, but the brief forbids external services |
| Own failover loop vs Router | A bit more code to own | Small, fully unit-tested; still uses LiteLLM errors + cache |
| Server-side history window | Very long chats lose early turns | `MAX_HISTORY_MESSAGES` configurable; summarisation later |
| Exact-match cache | Paraphrases miss | Semantic cache is out of scope (needs embeddings store) |
| Free-tier hosting cold starts | First request slow | `/health` warm-up; documented in README |

---

## 10. Testing strategy

- **Unit/integration (pytest + FastAPI TestClient)**, no real keys required: LiteLLM's
  `mock_response` path is used so the *real* LiteLLM cache is exercised.
- Covered: health, validation (422 cases), multi-turn history persistence, session
  isolation, cache hit/miss + identical reply + null `key_id`, failover via
  `X-Simulate-Failure` (`key_1=429`, `key_1=500`, `*=429` → 503 with no history write),
  404 on unknown session, header ignored when the flag is off, JSON-only errors.
- **Manual / demo:** curl scripts in the README against the live URLs.
