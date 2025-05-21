# prompt_optimizer/storage/local_storage.py

"""Local file-based storage implementation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from prompt_optimizer.storage.base import BaseStorage

T = TypeVar('T', bound=BaseModel)


class LocalStorage(BaseStorage[T]):
    """Local JSON file-based storage implementation."""

    def __init__(self, model_class: Type[T], storage_dir: str = "./data"):
        """Initialize local storage for a specific model type."""
        self.model_class = model_class
        self.storage_dir = Path(storage_dir)
        self.storage_file = self.storage_dir / f"{model_class.__name__.lower()}s.json"
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create storage file if it doesn't exist
        if not self.storage_file.exists():
            with open(self.storage_file, 'w') as f:
                json.dump({}, f)

    def _read_all(self) -> Dict[str, Dict]:
        """Read all items from storage."""
        with open(self.storage_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _write_all(self, data: Dict[str, Dict]) -> None:
        """Write all items to storage."""
        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save(self, item: T) -> T:
        """Save an item to storage."""
        data = self._read_all()
        # Update to use model_dump instead of dict
        data[item.id] = item.model_dump()
        self._write_all(data)
        return item

    def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID."""
        data = self._read_all()
        if item_id in data:
            # Update to use model_validate instead of parse_obj
            return self.model_class.model_validate(data[item_id])
        return None

    def update(self, item: T) -> T:
        """Update an existing item."""
        data = self._read_all()
        if item.id not in data:
            raise KeyError(f"Item with ID {item.id} does not exist")
        # Update to use model_dump instead of dict
        data[item.id] = item.model_dump()
        self._write_all(data)
        return item

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        data = self._read_all()
        if item_id not in data:
            return False
        del data[item_id]
        self._write_all(data)
        return True

    def list(self, filters: Optional[Dict] = None) -> List[T]:
        """List items, optionally filtered."""
        data = self._read_all()
        # Update to use model_validate instead of parse_obj
        result = [self.model_class.model_validate(item) for item in data.values()]
        
        if filters:
            filtered_result = []
            for item in result:
                matches = True
                for key, value in filters.items():
                    # Fix the filtering logic to properly match attributes
                    if hasattr(item, key):
                        if getattr(item, key) != value:
                            matches = False
                            break
                    else:
                        matches = False
                        break
                if matches:
                    filtered_result.append(item)
            return filtered_result
        
        return result