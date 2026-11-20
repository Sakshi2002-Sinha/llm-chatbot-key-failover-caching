# LLM Chatbot with Key Failover and Response Caching

A small, production-shaped AI chat app: **React** frontend, **FastAPI** backend, **LiteLLM**
gateway with a multi-key pool and automatic failover, LiteLLM's built-in **response cache**, and
**SQLite** session history. Every reply shows where it came from (`LIVE_INFERENCE` / `LLM_CACHE`),
the model, latency and the tokens used by *that* request.

📐 Full engineering design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

```
React (Vite, nginx :3000) ──▶ FastAPI (:8000) ──▶ KeyPool failover ──▶ LiteLLM acompletion ──▶ provider
                                   │                                      └─ LiteLLM cache (model + messages)
                                   └──▶ SQLite /data/app.db (Docker volume)
```

---

## 1. Live URLs

| Tier | URL |
|------|-----|
| Frontend | `https://<your-frontend>.vercel.app` ← *fill in after deploying* |
| Backend  | `https://<your-backend>.onrender.com` ← *fill in after deploying* |
| Health   | `https://<your-backend>.onrender.com/health` |
| Demo video | `<link>` |

### How to deploy (≈10 minutes)

**Backend → Render (Docker)**
1. Push this repo to GitHub (public).
2. Render → **New → Blueprint** → select the repo (uses [`render.yaml`](render.yaml)), *or*
   **New → Web Service** → Docker, Root Directory `backend`.
3. Environment variables:
   - `LLM_API_KEYS` = `key1,key2` (two real keys, e.g. two Google AI Studio Gemini keys)
   - `LLM_MODEL` = `gemini/gemini-3.5-flash-lite` (any LiteLLM model name works)
   - `ALLOWED_ORIGINS` = your Vercel URL, e.g. `https://my-chat.vercel.app`
   - `ENABLE_FAILURE_SIMULATION` = `true`
   - `DATABASE_URL` = `sqlite:////data/app.db`
4. Deploy and open `/health` → `{"status":"ok","keys_configured":2,"cache_enabled":true}`.

**Frontend → Vercel**
1. Vercel → **Add New Project** → import the repo, **Root Directory = `frontend`** (Vite preset).
2. Environment variable `VITE_API_BASE_URL` = your Render URL (no trailing slash).
3. Deploy. If you change the Vercel URL later, update `ALLOWED_ORIGINS` on Render.

> Render's free tier sleeps after inactivity, so the first request after a pause can take
> ~30–60 s. Open `/health` once before a demo.

---

## 2. Run locally

### With Docker (recommended)

```bash
cp .env.example .env          # then put two real keys in LLM_API_KEYS
make up                       # builds + starts both services
```

- Frontend: http://localhost:3000
- Backend:  http://localhost:8000 (`/health`, `/docs` for Swagger UI)
- `make down` stops everything; the `chat-data` volume (SQLite at `/data/app.db`) is kept, so
  history survives restarts. `make clean` also deletes the volume. `make logs` tails logs,
  `make test` runs the backend tests in a container.

> **Windows without `make`:** `docker compose up --build -d` / `docker compose down` do the same.

### Without Docker

Backend (Python 3.11+):

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
# set env vars (or export them from ../.env):
#   LLM_API_KEYS=key1,key2  ALLOWED_ORIGINS=http://localhost:3000
#   ENABLE_FAILURE_SIMULATION=true  DATABASE_URL=sqlite:///./data/app.db
uvicorn app.main:app --reload --port 8000
python -m pytest -q            # 43 tests, no real keys or network needed
```

Frontend (Node 18+):

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev    # http://localhost:3000
```

**No API keys handy?** Set `LLM_MOCK_RESPONSE="any text"` on the backend (with two dummy keys).
LiteLLM's mock provider then answers offline, while caching, failover, tokens and history all
run through the real code paths. Never set this in production.

---

## 3. Design decisions

### Key fallback

- `LLM_API_KEYS` is parsed into an ordered pool: `key_1`, `key_2`, … The raw key never leaves
  the `KeyPool` class. Responses and logs only ever contain the id.
- For each request, keys are tried in order. Each attempt is one `litellm.acompletion(...)` call
  with that key and `num_retries=0`, so failover is immediate and `fallback_attempts` is exact.
- Any key-level failure moves on to the next key: **429 / quota** (`RateLimitError`), **5xx**
  (`InternalServerError`, `ServiceUnavailableError`), connection errors and timeouts, auth errors
  (a revoked key), and empty replies. We stop early only for errors caused by the *input*
  (context window, content policy), since those would fail on every key.
- If every key fails, the API returns **`503 {"detail":"all providers unavailable"}`**. History
  is written only after a successful reply, in a single transaction, so a 503 (or 422) never
  leaves a half-written turn.
- **Cooldown (bonus):** a real 429/5xx puts that key in a short cooldown
  (`KEY_COOLDOWN_SECONDS`, default 30 s). Cooling keys go to the back of the order, so the next
  request doesn't waste a round-trip on a key known to be throttled. They're still tried as a
  last resort. Simulated failures never cause cooldown, because the hook is scoped to one request.
- **Why not `litellm.Router`?** The contract needs exact per-request attribution (`key_id`,
  `fallback_attempts`) and per-key, per-request failure injection. The Router's
  shuffle/retry/cooldown logic is opaque for that, so a ~60-line loop sits on top. LiteLLM still
  does everything provider-specific: the unified API, exception mapping, usage and caching.
- `LLM_MODELS` (optional) assigns a different model per key, e.g. a Gemini key and a Groq key.

### Caching

- LiteLLM's built-in in-memory cache: `litellm.cache = Cache(type="local")`.
- **Cache key = model + the full message list** (system prompt + history + new message). A
  custom key function hashes exactly that (SHA-256 of canonical JSON). It deliberately ignores
  the API key, so a reply cached via `key_1` is still a hit when `key_1` is down and `key_2`
  serves the request.
- Hits are detected from LiteLLM's own flag (`response._hidden_params["cache_hit"]`) and
  reported as `LLM_CACHE`, with `key_id: null` and `fallback_attempts: 0`. Token counts are the
  usage stored with the cached response, and the reply text is identical.
- **Two fixes found while testing:**
  1. LiteLLM writes the cache from a *background task*, so a request sent milliseconds after an
     identical one could still miss. After every live response we `await` the same cache write
     explicitly, which makes "the second identical request is `LLM_CACHE`" deterministic.
  2. LiteLLM lazily imports a large chunk of code on the first cache hit (~250 ms). The app warms
     both the live and cache-hit paths at startup with an offline mock call, so even the first
     real hit takes a few ms and is always faster than a live call.
- Why in-memory: the brief rules out Redis, and LiteLLM 1.110 no longer ships its `disk`
  backend. The trade-off is that the cache resets on restart and isn't shared between
  instances, which is fine for a single-instance deployment.

### SQLite session storage

- Path comes from `DATABASE_URL` (`sqlite:////data/app.db` → `/data/app.db`, a Docker named
  volume). Schema is created on startup. WAL mode is on.
- Tables: `messages` (the required schema, plus a `CHECK` on `role` and an index on
  `(session_id, id)`) and `sessions` (`session_id`, `system_prompt`, timestamps).
- The system prompt lives on `sessions`, not in `messages`. That way it can be set once and
  remembered for the session, and `GET /sessions/{id}/messages` naturally excludes it.
- The client sends only the new message. The server rebuilds context from SQLite (last
  `MAX_HISTORY_MESSAGES`). A per-session `asyncio.Lock` serialises concurrent turns of the same
  session so history order stays consistent.
- A session exists once it has a successful turn. Unknown (or only-failed) sessions → `404`.

### Error hygiene

Global exception handlers make every error JSON with a `detail` field: 404/405 routes, 422
validation (user input is never echoed back), 503 for providers, and a generic 500 for anything
unexpected. Stack traces go to the server log only.

---

## 4. Verification

Set `B` to the backend URL (`http://localhost:8000` or the Render URL).

```bash
B=http://localhost:8000

# Health
curl -s $B/health
# → {"status":"ok","keys_configured":2,"cache_enabled":true}

# Live inference
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}'
# → metadata.resolution_source = LIVE_INFERENCE, key_id = key_1

# Cache hit: same first message in a NEW session
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}'
# → LLM_CACHE, identical reply, key_id = null, lower latency_ms, same tokens

# Failover: key_1 rate-limited for this request only
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-demo-3","message":"Give me one fun fact about owls"}'
# → 200, key_id = key_2, fallback_attempts = 1   (key_1=500 behaves the same)

# Every key fails → clean 503, nothing stored
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-demo-4","message":"hello"}'
# → HTTP 503 {"detail":"all providers unavailable"};  GET /sessions/s-demo-4/messages → 404

# Multi-turn memory + history endpoint
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-mem","message":"My name is Asha."}'
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-mem","message":"What is my name?"}'
curl -s $B/sessions/s-mem/messages

# Validation → 422
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-x","message":"   "}'
```

PowerShell equivalent of a request:

```powershell
Invoke-RestMethod -Method Post -Uri "$env:B/chat" -ContentType 'application/json' -Headers @{ 'X-Simulate-Failure' = 'key_1=429' } -Body '{"session_id":"s-ps","message":"hi"}'
```

**In the UI**

- Every assistant bubble shows `source`, `model`, `latency`, `in`/`out` tokens, `key` and
  `fallbacks`. Tokens are per request: watch `in` grow as history grows, while `out` reflects
  only that reply.
- Cache hit: ask something, reload the page (a new session), and ask the identical question.
  The badge turns cyan `LLM_CACHE`, with `key null` and a lower latency.
- Failover: **Settings → Simulate failure → `key_1 → 429`**, then send. The reply arrives with
  `key key_2` and `fallbacks 1`. Choose `* → 429` to see the readable 503 message in the chat.
- **Settings → Reload history from server** re-renders the chat from
  `GET /sessions/{id}/messages`, which shows the history really lives in SQLite.

**Automated:** `cd backend && python -m pytest -q` runs 43 tests covering the contract, 422
cases, multi-turn and isolation, persistence across app restarts, cache hit/miss and latency,
cache sharing across keys, simulated and *real* (mocked LiteLLM exception) failover, cooldown,
503 without a history write, and CORS.

---

## 5. Setup Used

- **Editor:** VS Code / Claude Code desktop app on Windows 11
- **AI tools:** Claude Code (Claude Opus 5.5) for scaffolding, implementation, tests and docs.
  Every behaviour was verified by running the test suite, the Docker stack and the UI.
- **Runtime:** Python 3.11, FastAPI, LiteLLM 1.110, SQLite, Node 20 / Vite 5 / React 18, Docker
  Compose

---

## 6. Bonus features

| Feature | Why |
|---------|-----|
| **Failure simulator in the UI** (`data-testid="simulate-failure-select"`) | Demo failover and the 503 path without curl |
| **Key cooldown after a real 429/5xx** | Doesn't waste latency on a key known to be throttled; still uses it as a last resort |
| **Cross-key cache sharing** | Cache key ignores the API key, so cached answers survive a key outage |
| **Deterministic cache write + startup warm-up** | "Second identical request is a hit" and "hit is faster than live" hold even back-to-back and right after boot |
| **Per-session system prompt, remembered server-side** (`system-prompt-input`) | Set once and it persists for the session; excluded from history |
| **Per-key models (`LLM_MODELS`)** | Mix providers/models in one pool, e.g. a Gemini key and a Groq key |
| **Health badge, "New chat", "Reload history from server", char counter** | Shows backend state, keys and cache at a glance |
| **Offline mock mode (`LLM_MOCK_RESPONSE`)** | Tests and local demos run with zero keys through the real LiteLLM code paths |
| **Atomic turn writes + per-session lock** | No half-written turns, and history order stays consistent under concurrent requests |
| **Hardened containers** | Multi-stage builds, non-root backend user, Docker `HEALTHCHECK`s, `PORT` support for PaaS |

---

## Repository layout

```
backend/            FastAPI app (app/), pytest suite (tests/), multi-stage Dockerfile
frontend/           React + Vite SPA, nginx runtime image, vercel.json
docs/ARCHITECTURE.md  Engineering design doc
docker-compose.yml  backend :8000 + frontend :3000, named volume chat-data → /data
Makefile            up / down / logs / test / clean
render.yaml         Render blueprint for the backend
.env.example        Every variable with placeholder values
```
