# tests/test_storage.py

"""Test storage implementations."""

import os
import tempfile
from pathlib import Path

import pytest

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.storage.local_storage import LocalStorage


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_local_storage_save_and_get(temp_storage_dir):
    """Test saving and retrieving an item."""
    storage = LocalStorage(model_class=Prompt, storage_dir=temp_storage_dir)
    prompt = Prompt(text="Test prompt")
    
    # Save the prompt
    saved_prompt = storage.save(prompt)
    assert saved_prompt.id == prompt.id
    
    # Get the prompt
    retrieved_prompt = storage.get(prompt.id)
    assert retrieved_prompt is not None
    assert retrieved_prompt.id == prompt.id
    assert retrieved_prompt.text == "Test prompt"


def test_local_storage_update(temp_storage_dir):
    """Test updating an item."""
    storage = LocalStorage(model_class=Prompt, storage_dir=temp_storage_dir)
    prompt = Prompt(text="Original text")
    
    # Save the prompt
    storage.save(prompt)
    
    # Update the prompt
    prompt.text = "Updated text"
    updated_prompt = storage.update(prompt)
    
    # Verify the update
    assert updated_prompt.text == "Updated text"
    retrieved_prompt = storage.get(prompt.id)
    assert retrieved_prompt is not None
    assert retrieved_prompt.text == "Updated text"


def test_local_storage_delete(temp_storage_dir):
    """Test deleting an item."""
    storage = LocalStorage(model_class=Prompt, storage_dir=temp_storage_dir)
    prompt = Prompt(text="To be deleted")
    
    # Save the prompt
    storage.save(prompt)
    
    # Delete the prompt
    result = storage.delete(prompt.id)
    assert result is True
    
    # Verify it's gone
    assert storage.get(prompt.id) is None


def test_local_storage_list(temp_storage_dir):
    """Test listing items."""
    storage = LocalStorage(model_class=Prompt, storage_dir=temp_storage_dir)
    
    # Save multiple prompts
    prompt1 = Prompt(text="Prompt 1", description="Description 1")
    prompt2 = Prompt(text="Prompt 2", description="Description 2")
    prompt3 = Prompt(text="Prompt 3", description="Description 1")
    
    storage.save(prompt1)
    storage.save(prompt2)
    storage.save(prompt3)
    
    # List all prompts
    all_prompts = storage.list()
    assert len(all_prompts) == 3
    
    # Filter prompts
    filtered_prompts = storage.list(filters={"description": "Description 1"})
    assert len(filtered_prompts) == 2