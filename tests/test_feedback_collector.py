"""Tests for FeedbackCollector functionality."""

import pytest

from prompt_optimizer.core.models import Feedback, Response
from prompt_optimizer.core.feedback_collector import FeedbackCollector
from prompt_optimizer.storage.local_storage import LocalStorage


@pytest.fixture
def feedback_collector(tmp_path):
    """Create a FeedbackCollector with temporary storage."""
    feedback_storage = LocalStorage(model_class=Feedback, storage_dir=str(tmp_path))
    response_storage = LocalStorage(model_class=Response, storage_dir=str(tmp_path))

    # Create a test response
    response = Response(
        id="test-response-id",
        prompt_instance_id="test-instance-id",
        content="This is a test response",
    )
    response_storage.save(response)

    return FeedbackCollector(feedback_storage, response_storage)


def test_record_feedback(feedback_collector):
    """Test recording feedback for a response."""
    feedback = feedback_collector.record_feedback(
        response_id="test-response-id",
        score=0.9,
    )

    assert feedback.id is not None
    assert feedback.response_id == "test-response-id"
    assert feedback.score == 0.9


def test_get_feedback_for_response(feedback_collector):
    """Test retrieving feedback for a response."""
    feedback_collector.record_feedback(
        response_id="test-response-id",
        score=0.8,
    )
    feedback_collector.record_feedback(
        response_id="test-response-id",
        score=0.2,
    )

    feedback_items = feedback_collector.get_feedback_for_response("test-response-id")
    assert len(feedback_items) == 2
    assert any(item.score > 0 for item in feedback_items)


def test_record_feedback_invalid_response(feedback_collector):
    """Test recording feedback for an invalid response."""
    with pytest.raises(ValueError):
        feedback_collector.record_feedback(
            response_id="non-existent-id",
            score=0.5,
        )
