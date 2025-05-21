# prompt_optimizer/storage/base.py

"""Base class for storage implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, TypeVar, Generic

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class BaseStorage(Generic[T], ABC):
    """Abstract base class for storage implementations."""

    @abstractmethod
    def save(self, item: T) -> T:
        """Save an item to storage."""
        pass

    @abstractmethod
    def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        pass

    @abstractmethod
    def update(self, item: T) -> T:
        """Update an existing item."""
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        pass

    @abstractmethod
    def list(self, filters: Optional[Dict] = None) -> List[T]:
        """List items, optionally filtered."""
        pass