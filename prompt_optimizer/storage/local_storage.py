# prompt_optimizer/storage/local_storage.py

"""Local file-based storage implementation."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from prompt_optimizer.storage.base import BaseStorage
from prompt_optimizer.exceptions import StorageError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LocalStorage(BaseStorage[T]):
    """Local JSON file-based storage implementation."""

    def __init__(self, model_class: Type[T], storage_dir: str = "./data"):
        """Initialize local storage for a specific model type.
        
        Args:
            model_class: The Pydantic model class to store
            storage_dir: Directory for storage files
            
        Raises:
            StorageError: If initialization fails
        """
        try:
            self.model_class = model_class
            self.storage_dir = Path(storage_dir)
            self.storage_file = self.storage_dir / f"{model_class.__name__.lower()}s.json"
            
            # Create storage directory if it doesn't exist
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage directory created/verified: {self.storage_dir}")
            
            # Create storage file if it doesn't exist
            if not self.storage_file.exists():
                self._write_all({})
                logger.info(f"Created new storage file: {self.storage_file}")
            else:
                # Validate existing file can be read
                self._read_all()
                logger.info(f"Using existing storage file: {self.storage_file}")
                
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise StorageError(f"Storage initialization failed: {str(e)}") from e

    def _read_all(self) -> Dict[str, Dict]:
        """Read all items from storage.
        
        Returns:
            Dictionary mapping IDs to item data
            
        Raises:
            StorageError: If reading fails
        """
        try:
            if not self.storage_file.exists():
                logger.warning(f"Storage file does not exist: {self.storage_file}")
                return {}
                
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.info("Storage file is empty, returning empty dict")
                    return {}
                    
                data = json.loads(content)
                if not isinstance(data, dict):
                    logger.warning("Storage file contains non-dict data, resetting to empty")
                    return {}
                    
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON in storage file {self.storage_file}: {str(e)}")
            # Backup corrupted file
            backup_file = self.storage_file.with_suffix('.json.backup')
            try:
                self.storage_file.rename(backup_file)
                logger.info(f"Corrupted file backed up to: {backup_file}")
            except Exception as backup_error:
                logger.warning(f"Failed to backup corrupted file: {str(backup_error)}")
            
            # Return empty dict and let caller decide what to do
            return {}
            
        except PermissionError as e:
            logger.error(f"Permission denied reading storage file: {str(e)}")
            raise StorageError(f"Permission denied: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error reading storage: {str(e)}")
            raise StorageError(f"Failed to read storage: {str(e)}") from e

    def _write_all(self, data: Dict[str, Dict]) -> None:
        """Write all items to storage.
        
        Args:
            data: Dictionary mapping IDs to item data
            
        Raises:
            StorageError: If writing fails
        """
        try:
            # Validate data format
            if not isinstance(data, dict):
                raise ValidationError("Data must be a dictionary")
            
            # Create temporary file first for atomic write
            temp_file = self.storage_file.with_suffix('.json.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                f.flush()  # Ensure data is written
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic move from temp to actual file
            temp_file.replace(self.storage_file)
            logger.debug(f"Successfully wrote {len(data)} items to storage")
            
        except PermissionError as e:
            logger.error(f"Permission denied writing to storage: {str(e)}")
            raise StorageError(f"Permission denied: {str(e)}") from e
            
        except OSError as e:
            logger.error(f"OS error writing to storage: {str(e)}")
            raise StorageError(f"File system error: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error writing storage: {str(e)}")
            raise StorageError(f"Failed to write storage: {str(e)}") from e

    def save(self, item: T) -> T:
        """Save an item to storage.
        
        Args:
            item: Item to save
            
        Returns:
            The saved item
            
        Raises:
            StorageError: If saving fails
            ValidationError: If item is invalid
        """
        try:
            if not item:
                raise ValidationError("Item cannot be None")
                
            if not hasattr(item, 'id') or not item.id:
                raise ValidationError("Item must have a valid ID")
            
            data = self._read_all()
            data[item.id] = item.model_dump()
            self._write_all(data)
            
            logger.debug(f"Saved item with ID: {item.id}")
            return item
            
        except (ValidationError, StorageError):
            raise
        except Exception as e:
            logger.error(f"Failed to save item {getattr(item, 'id', 'unknown')}: {str(e)}")
            raise StorageError(f"Failed to save item: {str(e)}") from e

    def get(self, item_id: str) -> Optional[T]:
        """Get an item by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The item if found, None otherwise
            
        Raises:
            StorageError: If retrieval fails
            ValidationError: If item_id is invalid
        """
        try:
            if not item_id or not isinstance(item_id, str):
                raise ValidationError("Item ID must be a non-empty string")
            
            data = self._read_all()
            
            if item_id not in data:
                return None
            
            item_data = data[item_id]
            item = self.model_class.model_validate(item_data)
            
            logger.debug(f"Retrieved item with ID: {item_id}")
            return item
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get item {item_id}: {str(e)}")
            raise StorageError(f"Failed to retrieve item: {str(e)}") from e

    def update(self, item: T) -> T:
        """Update an existing item.
        
        Args:
            item: Item to update
            
        Returns:
            The updated item
            
        Raises:
            StorageError: If update fails
            ValidationError: If item doesn't exist or is invalid
        """
        try:
            if not item:
                raise ValidationError("Item cannot be None")
                
            if not hasattr(item, 'id') or not item.id:
                raise ValidationError("Item must have a valid ID")
            
            data = self._read_all()
            
            if item.id not in data:
                raise ValidationError(f"Item with ID {item.id} does not exist")
            
            data[item.id] = item.model_dump()
            self._write_all(data)
            
            logger.debug(f"Updated item with ID: {item.id}")
            return item
            
        except ValidationError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to update item {getattr(item, 'id', 'unknown')}: {str(e)}")
            raise StorageError(f"Failed to update item: {str(e)}") from e

    def delete(self, item_id: str) -> bool:
        """Delete an item by ID.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if item was deleted, False if not found
            
        Raises:
            StorageError: If deletion fails
            ValidationError: If item_id is invalid
        """
        try:
            if not item_id or not isinstance(item_id, str):
                raise ValidationError("Item ID must be a non-empty string")
            
            data = self._read_all()
            
            if item_id not in data:
                logger.info(f"Item {item_id} not found for deletion")
                return False
            
            del data[item_id]
            self._write_all(data)
            
            logger.debug(f"Deleted item with ID: {item_id}")
            return True
            
        except ValidationError:
            raise
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {str(e)}")
            raise StorageError(f"Failed to delete item: {str(e)}") from e

    def list(self, filters: Optional[Dict] = None) -> List[T]:
        """List items, optionally filtered.
        
        Args:
            filters: Optional dictionary of field filters
            
        Returns:
            List of matching items
            
        Raises:
            StorageError: If listing fails
        """
        try:
            data = self._read_all()
            
            # Convert all items to model instances
            items = []
            for item_id, item_data in data.items():
                try:
                    item = self.model_class.model_validate(item_data)
                    items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to parse item {item_id}: {str(e)}")
                    continue
            
            # Apply filters if provided
            if filters:
                filtered_items = []
                for item in items:
                    matches = True
                    for key, value in filters.items():
                        try:
                            if hasattr(item, key):
                                item_value = getattr(item, key)
                                if item_value != value:
                                    matches = False
                                    break
                            else:
                                matches = False
                                break
                        except Exception as e:
                            logger.warning(f"Error applying filter {key}={value}: {str(e)}")
                            matches = False
                            break
                    
                    if matches:
                        filtered_items.append(item)
                
                items = filtered_items
            
            logger.debug(f"Listed {len(items)} items (filters: {filters})")
            return items
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to list items: {str(e)}")
            raise StorageError(f"Failed to list items: {str(e)}") from e

    def count(self, filters: Optional[Dict] = None) -> int:
        """Count items, optionally filtered.
        
        Args:
            filters: Optional dictionary of field filters
            
        Returns:
            Number of matching items
            
        Raises:
            StorageError: If counting fails
        """
        try:
            items = self.list(filters)
            return len(items)
        except Exception as e:
            logger.error(f"Failed to count items: {str(e)}")
            raise StorageError(f"Failed to count items: {str(e)}") from e

    def clear(self) -> bool:
        """Clear all items from storage.
        
        Returns:
            True if successful
            
        Raises:
            StorageError: If clearing fails
        """
        try:
            self._write_all({})
            logger.info("Cleared all items from storage")
            return True
        except Exception as e:
            logger.error(f"Failed to clear storage: {str(e)}")
            raise StorageError(f"Failed to clear storage: {str(e)}") from e

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the storage file.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the backup file
            
        Raises:
            StorageError: If backup fails
        """
        try:
            if backup_path:
                backup_file = Path(backup_path)
            else:
                timestamp = str(int(time.time()))
                backup_file = self.storage_file.with_suffix(f'.backup.{timestamp}.json')
            
            if self.storage_file.exists():
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(self.storage_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
                return str(backup_file)
            else:
                raise StorageError("Storage file does not exist")
                
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise StorageError(f"Backup failed: {str(e)}") from e