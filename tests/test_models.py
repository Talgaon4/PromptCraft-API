"""Test the data models."""

import pytest
from prompt_optimizer.core.models import Prompt, PromptInstance, Response, Feedback


def test_prompt_creation():
    """Test creating a prompt."""
    prompt = Prompt(text="Summarize the following text: {input}")
    assert prompt.id is not None
    assert prompt.text == "Summarize the following text: {input}"
    assert prompt.version == 1


def test_prompt_instance_creation():
    """Test creating a prompt instance."""
    prompt = Prompt(text="Summarize the following text: {input}")
    instance = PromptInstance(
        prompt_id=prompt.id,
        formatted_text="Summarize the following text: This is a test."
    )
    assert instance.prompt_id == prompt.id
    assert instance.formatted_text == "Summarize the following text: This is a test."


def test_response_creation():
    """Test creating a response."""
    prompt = Prompt(text="Summarize the following text: {input}")
    instance = PromptInstance(
        prompt_id=prompt.id,
        formatted_text="Summarize the following text: This is a test."
    )
    response = Response(
        prompt_instance_id=instance.id,
        content="This is a test."
    )
    assert response.prompt_instance_id == instance.id
    assert response.content == "This is a test."


def test_feedback_creation():
    """Test creating feedback."""
    prompt = Prompt(text="Summarize the following text: {input}")
    instance = PromptInstance(
        prompt_id=prompt.id,
        formatted_text="Summarize the following text: This is a test."
    )
    response = Response(
        prompt_instance_id=instance.id,
        content="This is a test."
    )
    feedback = Feedback(
        response_id=response.id,
        score=0.9,
    )
    assert feedback.response_id == response.id
    assert feedback.score == 0.9
