# tests/test_prompt_manager.py

"""Tests for PromptManager functionality."""

import pytest

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.storage.local_storage import LocalStorage


@pytest.fixture
def prompt_manager(tmp_path):
    """Create a PromptManager with temporary storage."""
    storage = LocalStorage(model_class=Prompt, storage_dir=str(tmp_path))
    return PromptManager(storage)


def test_create_prompt(prompt_manager):
    """Test creating a new prompt."""
    prompt = prompt_manager.create_prompt(
        text="Analyze the following text: {input}",
        description="Text analysis prompt"
    )
    
    assert prompt.id is not None
    assert prompt.text == "Analyze the following text: {input}"
    assert prompt.description == "Text analysis prompt"
    assert prompt.version == 1
    assert prompt.parent_id is None


def test_get_prompt(prompt_manager):
    """Test retrieving a prompt."""
    created = prompt_manager.create_prompt("Test prompt")
    retrieved = prompt_manager.get_prompt(created.id)
    
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.text == created.text


def test_update_prompt(prompt_manager):
    """Test updating a prompt creates a new version."""
    original = prompt_manager.create_prompt("Original text")
    updated = prompt_manager.update_prompt(original.id, text="Updated text")
    
    assert updated.id != original.id  # New ID for new version
    assert updated.text == "Updated text"
    assert updated.version == 2
    assert updated.parent_id == original.id


def test_list_prompts(prompt_manager):
    """Test listing prompts with filtering."""
    prompt_manager.create_prompt("Prompt 1", "Category A")
    prompt_manager.create_prompt("Prompt 2", "Category B")
    prompt_manager.create_prompt("Prompt 3", "Category A")
    
    all_prompts = prompt_manager.list_prompts()
    assert len(all_prompts) == 3
    
    filtered_prompts = prompt_manager.list_prompts({"description": "Category A"})
    assert len(filtered_prompts) == 2


def test_get_prompt_history(prompt_manager):
    """Test retrieving version history of a prompt."""
    v1 = prompt_manager.create_prompt("Version 1")
    v2 = prompt_manager.update_prompt(v1.id, "Version 2")
    v3 = prompt_manager.update_prompt(v2.id, "Version 3")
    
    history = prompt_manager.get_prompt_history(v3.id)
    
    assert len(history) == 3
    assert history[0].text == "Version 1"
    assert history[1].text == "Version 2"
    assert history[2].text == "Version 3"