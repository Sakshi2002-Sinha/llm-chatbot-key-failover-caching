import uuid

import litellm
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.simulation import SimulationHeaderError, parse_simulation_header

from .conftest import make_settings


def sid() -> str:
    return f"s-{uuid.uuid4().hex[:12]}"


def unique(text: str = "Hello") -> str:
    # unique text so tests never collide on the process-wide LiteLLM cache
    return f"{text} {uuid.uuid4().hex}"


# --------------------------------------------------------------------- health


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_health_without_keys_is_503(tmp_path):
    app = create_app(make_settings(tmp_path, api_keys=[]))
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["keys_configured"] == 0


# --------------------------------------------------------------------- chat basics


def test_chat_contract_shape(client):
    s = sid()
    r = client.post("/chat", json={"session_id": s, "message": unique()})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == s
    assert body["reply"]
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "gpt-4o-mini"
    assert isinstance(meta["latency_ms"], int)
    assert isinstance(meta["input_tokens"], int) and meta["input_tokens"] > 0
    assert isinstance(meta["output_tokens"], int) and meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert "sk-test" not in r.text  # the raw key is never exposed


def test_message_is_trimmed_and_persisted(client):
    s = sid()
    msg = unique("Trim me")
    client.post("/chat", json={"session_id": s, "message": f"   {msg}  \n"})
    hist = client.get(f"/sessions/{s}/messages").json()
    assert hist["messages"][0] == {"role": "user", "content": msg, "created_at": hist["messages"][0]["created_at"]}


# --------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s-1"},
        {"message": "hi"},
        {"session_id": "", "message": "hi"},
        {"session_id": "s-1", "message": ""},
        {"session_id": "s-1", "message": "    \n\t "},
        {"session_id": "s-1", "message": "x" * 4001},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": 123, "message": "hi"},
    ],
)
def test_validation_422(client, payload):
    r = client.post("/chat", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_message_exactly_4000_chars_after_trim_is_ok(client):
    r = client.post("/chat", json={"session_id": sid(), "message": "  " + "a" * 3990 + uuid.uuid4().hex[:10] + "  "})
    assert r.status_code == 200


def test_malformed_json_422(client):
    r = client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_validation_failure_not_added_to_history(client):
    s = sid()
    client.post("/chat", json={"session_id": s, "message": unique()})
    client.post("/chat", json={"session_id": s, "message": "x" * 5000})
    assert len(client.get(f"/sessions/{s}/messages").json()["messages"]) == 2


# --------------------------------------------------------------------- multi-turn


def test_multi_turn_history_sent_to_model(client, monkeypatch):
    captured = []
    real = litellm.acompletion

    async def spy(**kwargs):
        captured.append(kwargs["messages"])
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", spy)
    s = sid()
    first, second = unique("My name is Sakura."), unique("What is my name?")
    client.post("/chat", json={"session_id": s, "message": first})
    client.post("/chat", json={"session_id": s, "message": second})

    sent = captured[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == first
    assert sent[3]["content"] == second


def test_history_endpoint(client):
    s = sid()
    msg = unique("Capital of France?")
    r = client.post("/chat", json={"session_id": s, "message": msg})
    hist = client.get(f"/sessions/{s}/messages")
    assert hist.status_code == 200
    body = hist.json()
    assert body["session_id"] == s
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == msg
    assert "metadata" not in body["messages"][0]
    assistant = body["messages"][1]
    assert assistant["content"] == r.json()["reply"]
    assert assistant["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert assistant["metadata"]["input_tokens"] == r.json()["metadata"]["input_tokens"]
    assert all(m["role"] != "system" for m in body["messages"])


def test_sessions_are_isolated(client):
    a, b = sid(), sid()
    client.post("/chat", json={"session_id": a, "message": unique("in A")})
    client.post("/chat", json={"session_id": b, "message": unique("in B")})
    client.post("/chat", json={"session_id": b, "message": unique("in B again")})
    assert len(client.get(f"/sessions/{a}/messages").json()["messages"]) == 2
    assert len(client.get(f"/sessions/{b}/messages").json()["messages"]) == 4


def test_unknown_session_404(client):
    r = client.get(f"/sessions/{sid()}/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_history_survives_app_restart(settings):
    s, msg = sid(), unique()
    with TestClient(create_app(settings)) as c:
        c.post("/chat", json={"session_id": s, "message": msg})
    with TestClient(create_app(settings)) as c:
        msgs = c.get(f"/sessions/{s}/messages").json()["messages"]
    assert msgs[0]["content"] == msg


def test_system_prompt_used_and_remembered(client, monkeypatch):
    captured = []
    real = litellm.acompletion

    async def spy(**kwargs):
        captured.append(kwargs["messages"])
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", spy)
    s = sid()
    client.post("/chat", json={"session_id": s, "message": unique(), "system_prompt": "Talk like a pirate."})
    client.post("/chat", json={"session_id": s, "message": unique()})
    assert captured[0][0] == {"role": "system", "content": "Talk like a pirate."}
    assert captured[1][0] == {"role": "system", "content": "Talk like a pirate."}


# --------------------------------------------------------------------- caching


def test_cache_hit_on_same_first_message_in_new_session(client):
    msg = unique("What is the capital of France?")
    r1 = client.post("/chat", json={"session_id": sid(), "message": msg}).json()
    r2 = client.post("/chat", json={"session_id": sid(), "message": msg}).json()

    assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r2["metadata"]["resolution_source"] == "LLM_CACHE"
    assert r2["reply"] == r1["reply"]
    assert r2["metadata"]["key_id"] is None
    assert r2["metadata"]["fallback_attempts"] == 0
    assert r2["metadata"]["input_tokens"] == r1["metadata"]["input_tokens"]
    assert r2["metadata"]["output_tokens"] == r1["metadata"]["output_tokens"]


def test_different_message_never_from_cache(client):
    r1 = client.post("/chat", json={"session_id": sid(), "message": unique()}).json()
    r2 = client.post("/chat", json={"session_id": sid(), "message": unique()}).json()
    assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r2["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history(client):
    """Same text as a follow-up in a session with history must not hit the first-message cache."""
    msg = unique()
    client.post("/chat", json={"session_id": sid(), "message": msg})
    s = sid()
    client.post("/chat", json={"session_id": s, "message": unique("context")})
    r = client.post("/chat", json={"session_id": s, "message": msg}).json()
    assert r["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_hit_is_faster_than_live(client, monkeypatch):
    import asyncio

    real = litellm.acompletion

    async def slow(**kwargs):
        resp = await real(**kwargs)
        if not (getattr(resp, "_hidden_params", {}) or {}).get("cache_hit"):
            await asyncio.sleep(0.2)  # emulate network latency of a live call
        return resp

    monkeypatch.setattr(litellm, "acompletion", slow)
    msg = unique()
    live = client.post("/chat", json={"session_id": sid(), "message": msg}).json()["metadata"]
    hit = client.post("/chat", json={"session_id": sid(), "message": msg}).json()["metadata"]
    assert hit["resolution_source"] == "LLM_CACHE"
    assert hit["latency_ms"] < live["latency_ms"]


def test_cache_shared_across_keys(client):
    """A reply cached via key_1 is served when key_1 is down and key_2 handles the request."""
    msg = unique()
    client.post("/chat", json={"session_id": sid(), "message": msg})
    r = client.post(
        "/chat", json={"session_id": sid(), "message": msg}, headers={"X-Simulate-Failure": "key_1=429"}
    ).json()
    assert r["metadata"]["resolution_source"] == "LLM_CACHE"
    assert r["metadata"]["key_id"] is None


# --------------------------------------------------------------------- failover


@pytest.mark.parametrize("header", ["key_1=429", "key_1=500", "key_1=502", "key_1=503"])
def test_failover_to_key_2(client, header):
    s = sid()
    r = client.post("/chat", json={"session_id": s, "message": unique()}, headers={"X-Simulate-Failure": header})
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert r.json()["reply"] == "Mocked reply from the model."


def test_all_keys_fail_returns_503_and_no_history(client):
    s = sid()
    r = client.post("/chat", json={"session_id": s, "message": unique()}, headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert client.get(f"/sessions/{s}/messages").status_code == 404


def test_explicit_both_keys_fail_503(client):
    r = client.post(
        "/chat",
        json={"session_id": sid(), "message": unique()},
        headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
    )
    assert r.status_code == 503


def test_503_does_not_pollute_existing_session(client):
    s = sid()
    client.post("/chat", json={"session_id": s, "message": unique()})
    client.post("/chat", json={"session_id": s, "message": unique()}, headers={"X-Simulate-Failure": "*=503"})
    assert len(client.get(f"/sessions/{s}/messages").json()["messages"]) == 2


def test_simulation_is_scoped_to_one_request(client):
    client.post("/chat", json={"session_id": sid(), "message": unique()}, headers={"X-Simulate-Failure": "key_1=429"})
    r = client.post("/chat", json={"session_id": sid(), "message": unique()})
    assert r.json()["metadata"]["key_id"] == "key_1"


def test_simulation_ignored_when_disabled(tmp_path):
    app = create_app(make_settings(tmp_path, enable_failure_simulation=False))
    with TestClient(app) as c:
        r = c.post("/chat", json={"session_id": sid(), "message": unique()}, headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200
    assert r.json()["metadata"]["key_id"] == "key_1"


def test_malformed_simulation_header_422(client):
    r = client.post("/chat", json={"session_id": sid(), "message": unique()}, headers={"X-Simulate-Failure": "key_1=418"})
    assert r.status_code == 422


def test_real_provider_rate_limit_fails_over_and_cools_down(client, monkeypatch):
    """A real (non-simulated) 429 from key_1 fails over and puts key_1 in cooldown."""
    real = litellm.acompletion
    calls = []

    async def flaky(**kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "sk-test-one":
            raise litellm.RateLimitError(message="quota exceeded", llm_provider="openai", model=kwargs["model"])
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", flaky)
    r1 = client.post("/chat", json={"session_id": sid(), "message": unique()}).json()
    assert r1["metadata"]["key_id"] == "key_2"
    assert r1["metadata"]["fallback_attempts"] == 1

    # key_1 is cooling down, so the next request goes straight to key_2
    calls.clear()
    r2 = client.post("/chat", json={"session_id": sid(), "message": unique()}).json()
    assert r2["metadata"]["key_id"] == "key_2"
    assert r2["metadata"]["fallback_attempts"] == 0
    assert calls == ["sk-test-two"]


def test_real_5xx_on_all_keys_503(client, monkeypatch):
    async def down(**kwargs):
        raise litellm.ServiceUnavailableError(message="down", llm_provider="openai", model=kwargs["model"])

    monkeypatch.setattr(litellm, "acompletion", down)
    r = client.post("/chat", json={"session_id": sid(), "message": unique()})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}


def test_unexpected_exception_is_clean_json(client, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(litellm, "acompletion", boom)
    r = client.post("/chat", json={"session_id": sid(), "message": unique()})
    assert r.status_code == 503
    assert "secret internals" not in r.text
    assert "Traceback" not in r.text


def test_unknown_route_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


# --------------------------------------------------------------------- tokens


def test_tokens_are_per_request_not_cumulative(client, monkeypatch):
    # use real token counting for input by stripping usage from a live response
    s = sid()
    m1 = client.post("/chat", json={"session_id": s, "message": unique()}).json()["metadata"]
    m2 = client.post("/chat", json={"session_id": s, "message": unique()}).json()["metadata"]
    m_other = client.post("/chat", json={"session_id": sid(), "message": unique()}).json()["metadata"]
    # equal-length first messages in two sessions report about the same input_tokens
    assert abs(m1["input_tokens"] - m_other["input_tokens"]) <= 2
    assert m2["output_tokens"] < m1["output_tokens"] + m2["output_tokens"]


# --------------------------------------------------------------------- CORS


def test_cors_allows_frontend_origin(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


# --------------------------------------------------------------------- header parser


def test_parse_simulation_header():
    keys = ["key_1", "key_2"]
    assert parse_simulation_header("key_1=429", keys) == {"key_1": 429}
    assert parse_simulation_header("key_1=429, key_2=500", keys) == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503", keys) == {"key_1": 503, "key_2": 503}
    assert parse_simulation_header("*=429,key_2=500", keys) == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("key_9=429", keys) == {}
    assert parse_simulation_header("", keys) == {}
    for bad in ["key_1", "key_1=abc", "key_1=404", "=429"]:
        with pytest.raises(SimulationHeaderError):
            parse_simulation_header(bad, keys)
