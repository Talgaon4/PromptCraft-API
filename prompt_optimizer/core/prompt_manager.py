# prompt_optimizer/core/prompt_manager.py

"""Manages prompts and their versions."""

from typing import Dict, List, Optional, Any

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.storage.base import BaseStorage


class PromptManager:
    """Manages the creation, retrieval, and versioning of prompts."""

    def __init__(self, storage: BaseStorage[Prompt]):
        """Initialize with a storage implementation."""
        self.storage = storage

    def create_prompt(self, text: str, description: str = "") -> Prompt:
        """Create a new prompt."""
        prompt = Prompt(text=text, description=description)
        return self.storage.save(prompt)

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by ID."""
        return self.storage.get(prompt_id)

    def update_prompt(self, prompt_id: str, text: str = None, description: str = None) -> Prompt:
        """Update an existing prompt, creating a new version."""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt with ID {prompt_id} not found")

        # Create a new version with the old prompt as parent
        new_prompt = Prompt(
            text=text if text is not None else prompt.text,
            description=description if description is not None else prompt.description,
            version=prompt.version + 1,
            parent_id=prompt.id
        )
        return self.storage.save(new_prompt)

    def list_prompts(self, filters: Optional[Dict[str, Any]] = None) -> List[Prompt]:
        """List prompts with optional filtering."""
        return self.storage.list(filters)

    def get_prompt_history(self, prompt_id: str) -> List[Prompt]:
        """Get the version history of a prompt."""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt with ID {prompt_id} not found")

        history = [prompt]
        current_id = prompt.parent_id

        while current_id:
            parent = self.get_prompt(current_id)
            if parent:
                history.append(parent)
                current_id = parent.parent_id
            else:
                break

        # Return in chronological order (oldest first)
        return list(reversed(history))