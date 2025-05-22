# prompt_optimizer/exceptions.py

"""
Custom exceptions for PromptCraft API.
"""

class PromptCraftError(Exception):
    """Base exception for all PromptCraft errors."""
    pass


class PromptNotFoundError(PromptCraftError):
    """Raised when a prompt ID doesn't exist."""
    pass


class ResponseNotFoundError(PromptCraftError):
    """Raised when a response ID doesn't exist."""
    pass


class OptimizationError(PromptCraftError):
    """Raised when optimization fails."""
    pass


class LLMError(PromptCraftError):
    """Raised when LLM service fails."""
    pass


class ValidationError(PromptCraftError):
    """Raised when input validation fails."""
    pass


class StorageError(PromptCraftError):
    """Raised when storage operations fail."""
    pass


# Simple helper function for common validation
def validate_not_empty(value, name):
    """Simple validation helper."""
    if not value or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{name} cannot be empty")
    return value.strip() if isinstance(value, str) else value


def validate_prompt_id(prompt_id):
    """Validate prompt ID format."""
    validate_not_empty(prompt_id, "Prompt ID")
    return prompt_id