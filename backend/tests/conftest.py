import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Route LiteLLM through its built-in mock provider: no network, no real keys,
# but the real LiteLLM cache and response objects are exercised.
os.environ.setdefault("LLM_MOCK_RESPONSE", "Mocked reply from the model.")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        api_keys=["sk-test-one", "sk-test-two"],
        model="gpt-4o-mini",
        database_path=str(tmp_path / "app.db"),
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
