# prompt_optimizer/response_objects.py

"""
Standard response objects for consistent API returns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class OperationResult:
    """Standard response for all operations."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


@dataclass
class PromptResult(OperationResult):
    """Response for prompt operations."""
    prompt_id: Optional[str] = None
    prompt_text: Optional[str] = None
    version: Optional[int] = None
    
    @classmethod
    def success(cls, prompt_data: Dict[str, Any], message: str = "Operation successful") -> 'PromptResult':
        """Create a successful prompt result."""
        return cls(
            success=True,
            data=prompt_data,
            message=message,
            prompt_id=prompt_data.get('id'),
            prompt_text=prompt_data.get('text'),
            version=prompt_data.get('version')
        )
    
    @classmethod
    def failure(cls, message: str, errors: List[str] = None) -> 'PromptResult':
        """Create a failed prompt result."""
        return cls(
            success=False,
            message=message,
            errors=errors or []
        )


@dataclass
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
            success=True,
            message=message,
            original_prompt_id=original_id,
            new_prompt_id=new_id,
            optimization_applied=applied,
            improvement_reason=reason
        )
    
    @classmethod
    def not_ready(cls, 
                  prompt_id: str,
                  readiness_info: Dict[str, Any],
                  message: str = "Not ready for optimization") -> 'OptimizationResult':
        """Create a result for when optimization isn't ready."""
        return cls(
            success=False,
            message=message,
            original_prompt_id=prompt_id,
            readiness_info=readiness_info
        )
    
    @classmethod
    def failure(cls, prompt_id: str, message: str, errors: List[str] = None) -> 'OptimizationResult':
        """Create a failed optimization result."""
        return cls(
            success=False,
            message=message,
            errors=errors or [],
            original_prompt_id=prompt_id
        )


@dataclass
class ValidationResult(OperationResult):
    """Response for validation operations."""
    is_valid: bool = False
    validation_details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def valid(cls, details: Dict[str, Any] = None, message: str = "Validation passed") -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(
            success=True,
            is_valid=True,
            message=message,
            validation_details=details
        )
    
    @classmethod
    def invalid(cls, message: str, details: Dict[str, Any] = None, errors: List[str] = None) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(
            success=False,
            is_valid=False,
            message=message,
            validation_details=details,
            errors=errors or []
        )