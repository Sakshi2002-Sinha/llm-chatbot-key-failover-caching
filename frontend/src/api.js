// Thin API client. Every failure is normalised into an Error with a readable
// message so the UI can render it inside the chat instead of crashing.

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

function formatDetail(detail) {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI 422 shape: [{loc: [...], msg: "..."}]
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        const msg = (d.msg || "").replace(/^Value error, /, "");
        return field ? `${field}: ${msg}` : msg;
      })
      .join("; ");
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    const err = new Error("Cannot reach the server. Check your connection and try again.");
    err.status = 0;
    throw err;
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    const detail = formatDetail(body && body.detail) || res.statusText || "Request failed";
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return body;
}

export function sendChat({ sessionId, message, systemPrompt, simulateFailure }) {
  const headers = { "Content-Type": "application/json" };
  if (simulateFailure) headers["X-Simulate-Failure"] = simulateFailure;
  const payload = { session_id: sessionId, message };
  if (systemPrompt && systemPrompt.trim()) payload.system_prompt = systemPrompt.trim();
  return request("/chat", { method: "POST", headers, body: JSON.stringify(payload) });
}

export function getHealth() {
  return request("/health");
}

export function getHistory(sessionId) {
  return request(`/sessions/${encodeURIComponent(sessionId)}/messages`);
}
