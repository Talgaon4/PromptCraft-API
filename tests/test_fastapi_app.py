import pytest
from fastapi.testclient import TestClient

from fastapi_app import main
from prompt_optimizer.api import PromptOptimizer


@pytest.fixture
def client(tmp_path):
    # Replace the global optimizer with one using a temp directory
    original_optimizer = main.optimizer
    main.optimizer = PromptOptimizer(
        storage_dir=str(tmp_path), optimization_threshold=2
    )
    client = TestClient(main.app)
    yield client
    main.optimizer = original_optimizer


def test_fastapi_workflow(client):
    # Register a prompt
    resp = client.post(
        "/prompts", params={"text": "Echo {text}", "description": "test"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    prompt_id = data["prompt_id"]

    # Record usage
    resp = client.post(
        f"/prompts/{prompt_id}/use", params={"formatted_text": "Echo hello"}
    )
    assert resp.status_code == 200
    instance_id = resp.json()["data"]["instance_id"]

    # Record response
    resp = client.post(f"/responses/{instance_id}", params={"content": "hello"})
    assert resp.status_code == 200
    response_id = resp.json()["data"]["response_id"]

    # Record feedback
    resp = client.post(f"/feedback/{response_id}", params={"score": 0.9})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Get optimization stats
    resp = client.get(f"/optimization/{prompt_id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Force optimization
    resp = client.post(f"/optimize/{prompt_id}", params={"force": True})
    assert resp.status_code == 200
    assert "success" in resp.json()
