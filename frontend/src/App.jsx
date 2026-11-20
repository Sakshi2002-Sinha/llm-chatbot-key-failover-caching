import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL, getHealth, getHistory, sendChat } from "./api.js";

const MAX_CHARS = 4000;

const SIMULATION_OPTIONS = [
  { value: "", label: "None (normal)" },
  { value: "key_1=429", label: "key_1 → 429 (rate limit)" },
  { value: "key_1=500", label: "key_1 → 500 (server error)" },
  { value: "key_1=503", label: "key_1 → 503 (unavailable)" },
  { value: "key_1=429,key_2=500", label: "key_1 → 429, key_2 → 500" },
  { value: "*=429", label: "* → 429 (all keys fail)" },
];

function newSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `s-${crypto.randomUUID()}`;
  }
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

let localIdCounter = 0;
const localId = () => `m-${++localIdCounter}`;

function MetaItem({ label, testId, children, tone }) {
  return (
    <span className={`meta-item ${tone ? `meta-${tone}` : ""}`}>
      <span className="meta-label">{label}</span>
      <span className="meta-value" data-testid={testId}>
        {children}
      </span>
    </span>
  );
}

function AssistantMessage({ content, metadata }) {
  const m = metadata || {};
  const cached = m.resolution_source === "LLM_CACHE";
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant" data-testid="assistant-message">
        <div className="bubble-text">{content}</div>
        <div className="meta">
          <MetaItem label="source" testId="meta-source" tone={cached ? "cache" : "live"}>
            {m.resolution_source ?? "—"}
          </MetaItem>
          <MetaItem label="model" testId="meta-model">
            {m.model ?? "—"}
          </MetaItem>
          <MetaItem label="latency" testId="meta-latency">
            {m.latency_ms != null ? `${m.latency_ms} ms` : "—"}
          </MetaItem>
          <MetaItem label="in" testId="meta-input-tokens">
            {m.input_tokens ?? "—"}
          </MetaItem>
          <MetaItem label="out" testId="meta-output-tokens">
            {m.output_tokens ?? "—"}
          </MetaItem>
          {"key_id" in m && (
            <MetaItem label="key" testId="meta-key-id">
              {m.key_id ?? "null"}
            </MetaItem>
          )}
          {"fallback_attempts" in m && (
            <MetaItem
              label="fallbacks"
              testId="meta-fallback-attempts"
              tone={m.fallback_attempts > 0 ? "warn" : undefined}
            >
              {m.fallback_attempts}
            </MetaItem>
          )}
        </div>
      </div>
    </div>
  );
}

function UserMessage({ content }) {
  return (
    <div className="row row-user">
      <div className="bubble bubble-user" data-testid="user-message">
        <div className="bubble-text">{content}</div>
      </div>
    </div>
  );
}

function ErrorMessage({ content, status }) {
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-error" role="alert" data-testid="error-message">
        <strong>{status ? `Error ${status}` : "Error"}:</strong> {content}
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [simulateFailure, setSimulateFailure] = useState("");
  const [pending, setPending] = useState(false);
  const [health, setHealth] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const listRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealth({ status: "down", error: e.message }));
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, pending]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || pending) return;

    setMessages((prev) => [...prev, { id: localId(), role: "user", content: text }]);
    setInput("");
    setPending(true);
    try {
      const res = await sendChat({ sessionId, message: text, systemPrompt, simulateFailure });
      setMessages((prev) => [
        ...prev,
        { id: localId(), role: "assistant", content: res.reply, metadata: res.metadata },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: localId(), role: "error", content: err.message, status: err.status },
      ]);
    } finally {
      setPending(false);
      inputRef.current?.focus();
    }
  }, [input, pending, sessionId, systemPrompt, simulateFailure]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  const newChat = () => {
    setSessionId(newSessionId());
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  };

  const reloadHistory = async () => {
    try {
      const res = await getHistory(sessionId);
      setMessages(res.messages.map((m) => ({ id: localId(), ...m })));
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: localId(), role: "error", content: err.message, status: err.status },
      ]);
    }
  };

  const healthy = health && health.status === "ok";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">◆</span>
          <div>
            <h1>Resilient LLM Chat</h1>
            <p className="subtitle">Key failover · response caching · per-request tokens</p>
          </div>
        </div>
        <div className="header-right">
          <span
            className={`status ${healthy ? "status-ok" : health ? "status-down" : ""}`}
            title={health?.error || API_BASE_URL}
          >
            <span className="dot" />
            {health == null
              ? "checking…"
              : healthy
              ? `${health.keys_configured} keys · cache ${health.cache_enabled ? "on" : "off"}`
              : "backend unavailable"}
          </span>
          <button className="btn btn-ghost" onClick={() => setShowSettings((s) => !s)}>
            {showSettings ? "Hide settings" : "Settings"}
          </button>
          <button className="btn btn-ghost" onClick={newChat}>
            New chat
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings">
          <label className="field">
            <span>System prompt (this session)</span>
            <textarea
              data-testid="system-prompt-input"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="You are a helpful assistant."
              rows={2}
              maxLength={4000}
            />
          </label>
          <div className="settings-row">
            <label className="field field-inline">
              <span>Simulate failure (X-Simulate-Failure)</span>
              <select
                data-testid="simulate-failure-select"
                value={simulateFailure}
                onChange={(e) => setSimulateFailure(e.target.value)}
              >
                {SIMULATION_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <button className="btn btn-ghost" onClick={reloadHistory} title="GET /sessions/{id}/messages">
              Reload history from server
            </button>
          </div>
          <p className="session-line">
            Session: <code data-testid="session-id">{sessionId}</code>
          </p>
        </section>
      )}

      <main className="chat" ref={listRef}>
        {messages.length === 0 && !pending && (
          <div className="empty">
            <h2>Start a conversation</h2>
            <p>
              Each reply shows where it came from (<code>LIVE_INFERENCE</code> or <code>LLM_CACHE</code>),
              the model, latency and the tokens used by that request.
            </p>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <UserMessage key={m.id} content={m.content} />
          ) : m.role === "assistant" ? (
            <AssistantMessage key={m.id} content={m.content} metadata={m.metadata} />
          ) : (
            <ErrorMessage key={m.id} content={m.content} status={m.status} />
          )
        )}
        {pending && (
          <div className="row row-assistant">
            <div className="bubble bubble-assistant typing" aria-label="Assistant is typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </main>

      <footer className="composer">
        {simulateFailure && (
          <div className="sim-banner">
            Simulating failure: <code>{simulateFailure}</code>
          </div>
        )}
        <div className="composer-row">
          <textarea
            ref={inputRef}
            data-testid="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
            rows={1}
            autoFocus
          />
          <button
            className="btn btn-primary"
            data-testid="send-button"
            onClick={send}
            disabled={pending || !input.trim()}
          >
            {pending ? "Sending…" : "Send"}
          </button>
        </div>
        <div className={`counter ${input.trim().length > MAX_CHARS ? "counter-over" : ""}`}>
          {input.trim().length} / {MAX_CHARS}
        </div>
      </footer>
    </div>
  );
}
