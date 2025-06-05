"""Data models for the Prompt Optimizer API."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


class Prompt(BaseModel):
    """Model representing a prompt template."""
    id: str = Field(default_factory=generate_id)
    text: str
    version: int = 1
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    parent_id: Optional[str] = None


class PromptInstance(BaseModel):
    """Model representing a specific instance of a prompt that was used."""
    id: str = Field(default_factory=generate_id)
    prompt_id: str
    formatted_text: str
    created_at: datetime = Field(default_factory=datetime.now)
    context: Dict = Field(default_factory=dict)


class Response(BaseModel):
    """Model representing a response received for a prompt instance."""
    id: str = Field(default_factory=generate_id)
    prompt_instance_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)


class Feedback(BaseModel):
    """Model representing feedback for a response."""

    id: str = Field(default_factory=generate_id)
    response_id: str
    score: float
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_positive(self) -> bool:
        """Return ``True`` if the score indicates positive feedback."""
        return self.score >= 0.5
