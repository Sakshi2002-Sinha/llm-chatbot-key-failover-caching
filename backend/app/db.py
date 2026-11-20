"""SQLite persistence for sessions and messages."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    metadata   TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self) -> None:
        if self.path != ":memory:":
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # -- reads ---------------------------------------------------------------

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            if row["role"] == "assistant" and row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            out.append(item)
        return out

    # -- writes --------------------------------------------------------------

    def save_turn(
        self,
        session_id: str,
        system_prompt: str | None,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a successful turn atomically (session upsert + both messages)."""
        now = utcnow_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt), "
                "updated_at = excluded.updated_at",
                (session_id, system_prompt, now, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_content, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_content, json.dumps(metadata), utcnow_iso()),
            )
