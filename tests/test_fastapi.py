
import pytest
from fastapi.testclient import TestClient

from fastapi_app.main import app
from fastapi_app.deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer


@pytest.fixture
def client(tmp_path):
    def override():
        return PromptOptimizer(storage_dir=str(tmp_path))
    app.dependency_overrides[get_optimizer] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_endpoints_flow(client):
    # register prompt
    r = client.post("/prompts", json={"text": "Hello {name}", "description": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["is_successful"] is True
    prompt_id = data["prompt_id"]

    # get prompt
    r = client.get(f"/prompts/{prompt_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["prompt_id"] == prompt_id

    # record usage
    r = client.post(f"/prompts/{prompt_id}/use", json={"formatted_text": "Hello Tom"})
    assert r.status_code == 200
    inst_id = r.json()["data"]["instance_id"]

    # record response
    r = client.post(f"/instances/{inst_id}/responses", json={"content": "hi"})
    assert r.status_code == 200
    resp_id = r.json()["data"]["response_id"]

    # record feedback
    r = client.post(f"/responses/{resp_id}/feedback", json={"score": 0.8})
    assert r.status_code == 200
    assert r.json()["is_successful"] is True

    # get stats
    r = client.get(f"/prompts/{prompt_id}/stats")
    assert r.status_code == 200
    assert "is_successful" in r.json()

    # optimize
    r = client.post(f"/prompts/{prompt_id}/optimize?force=true")
    assert r.status_code == 200
    assert "is_successful" in r.json()

    # config
    r = client.get("/config")
    assert r.status_code == 200
    assert r.json()["is_successful"] is True

