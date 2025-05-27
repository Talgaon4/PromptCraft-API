# prompt_optimizer/response_objects.py

"""
Standard response objects for consistent API returns.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OperationResult(BaseModel):
    """Standard response for all operations."""
    is_successful: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Property for backward compatibility
    @property
    def success(self) -> bool:
        """Backward compatibility property."""
        return self.is_successful
    
    class Config:
        # Allow arbitrary types for flexibility
        arbitrary_types_allowed = True
        # Use enum values for serialization
        use_enum_values = True
        # Allow JSON serialization of datetime
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


class PromptResult(OperationResult):
    """Response for prompt operations."""
    prompt_id: Optional[str] = None
    prompt_text: Optional[str] = None
    version: Optional[int] = Field(None, ge=1)  # Version must be >= 1 if provided
    
    @classmethod
    def success(cls, prompt_data: Dict[str, Any], message: str = "Operation successful") -> 'PromptResult':
        """Create a successful prompt result."""
        return cls(
            is_successful=True,
            data=prompt_data,
            message=message,
            prompt_id=prompt_data.get('id'),
            prompt_text=prompt_data.get('text'),
            version=prompt_data.get('version'),
            errors=[]
        )
    
    @classmethod
    def failure(cls, message: str, errors: List[str] = None) -> 'PromptResult':
        """Create a failed prompt result."""
        return cls(
            is_successful=False,
            message=message,
            errors=errors or [],
            prompt_id=None,
            prompt_text=None,
            version=None,
            data=None
        )


class OptimizationResult(OperationResult):
    """Response for optimization operations."""
    original_prompt_id: Optional[str] = None
    new_prompt_id: Optional[str] = None
    optimization_applied: bool = False
    improvement_reason: Optional[str] = None
    readiness_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success(cls, 
                original_id: str,
                new_id: str = None, 
                applied: bool = False,
                reason: str = None,
                message: str = "Optimization completed") -> 'OptimizationResult':
        """Create a successful optimization result."""
        return cls(
            is_successful=True,
            message=message,
            original_prompt_id=original_id,
            new_prompt_id=new_id,
            optimization_applied=applied,
            improvement_reason=reason,
            errors=[]
        )
    
    @classmethod
    def not_ready(cls, 
                  prompt_id: str,
                  readiness_info: Dict[str, Any],
                  message: str = "Not ready for optimization") -> 'OptimizationResult':
        """Create a result for when optimization isn't ready."""
        return cls(
            is_successful=False,
            message=message,
            original_prompt_id=prompt_id,
            readiness_info=readiness_info,
            errors=[]
        )
    
    @classmethod
    def failure(cls, prompt_id: str, message: str, errors: List[str] = None) -> 'OptimizationResult':
        """Create a failed optimization result."""
        return cls(
            is_successful=False,
            message=message,
            errors=errors or [],
            original_prompt_id=prompt_id,
            new_prompt_id=None,
            optimization_applied=False,
            improvement_reason=None,
            readiness_info=None
        )


class ValidationResult(OperationResult):
    """Response for validation operations."""
    is_valid: bool = False
    validation_details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def valid(cls, details: Dict[str, Any] = None, message: str = "Validation passed") -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(
            is_successful=True,
            is_valid=True,
            message=message,
            validation_details=details,
            errors=[]
        )
    
    @classmethod
    def invalid(cls, message: str, details: Dict[str, Any] = None, errors: List[str] = None) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(
            is_successful=False,
            is_valid=False,
            message=message,
            validation_details=details,
            errors=errors or []
        )