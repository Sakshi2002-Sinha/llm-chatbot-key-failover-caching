"""Parsing for the X-Simulate-Failure test hook."""
from __future__ import annotations

ALLOWED_STATUSES = {429, 500, 502, 503}


class SimulationHeaderError(ValueError):
    pass


def parse_simulation_header(raw: str | None, key_ids: list[str]) -> dict[str, int]:
    """Parse ``key_1=429,key_2=500`` / ``*=429`` into ``{key_id: status}``.

    ``*`` expands to every configured key; a later explicit entry overrides it.
    Unknown key ids are ignored (they cannot fail anything). Raises
    SimulationHeaderError on a malformed pair or unsupported status.
    """
    if raw is None or not raw.strip():
        return {}

    wildcard: int | None = None
    explicit: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SimulationHeaderError(f"invalid X-Simulate-Failure entry '{part}', expected <key_id>=<status>")
        key, _, status_raw = part.partition("=")
        key, status_raw = key.strip(), status_raw.strip()
        try:
            status = int(status_raw)
        except ValueError:
            raise SimulationHeaderError(f"invalid status '{status_raw}' in X-Simulate-Failure") from None
        if status not in ALLOWED_STATUSES:
            raise SimulationHeaderError(
                f"unsupported status {status} in X-Simulate-Failure; allowed: 429, 500, 502, 503"
            )
        if not key:
            raise SimulationHeaderError("empty key id in X-Simulate-Failure")
        if key == "*":
            wildcard = status
        else:
            explicit[key] = status

    result: dict[str, int] = {}
    if wildcard is not None:
        result = {k: wildcard for k in key_ids}
    for k, s in explicit.items():
        if k in key_ids:
            result[k] = s
    return result
