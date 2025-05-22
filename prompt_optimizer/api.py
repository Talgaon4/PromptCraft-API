# prompt_optimizer/api.py

"""Main API interface for the Prompt Optimizer."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from prompt_optimizer.core.models import Prompt, PromptInstance, Response, Feedback
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.response_tracker import ResponseTracker
from prompt_optimizer.core.feedback_collector import FeedbackCollector
from prompt_optimizer.core.optimizer_engine import OptimizerEngine
from prompt_optimizer.storage.local_storage import LocalStorage
from prompt_optimizer.config import config, create_config
from prompt_optimizer.exceptions import (
    PromptNotFoundError, ResponseNotFoundError, OptimizationError, 
    ValidationError, StorageError, validate_prompt_id, validate_not_empty
)

# Simple logging setup
logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Main API interface for the Prompt Optimizer."""

    def __init__(self, **kwargs):
        """Initialize the Prompt Optimizer API.
        
        Args:
            **kwargs: Configuration overrides
            
        Raises:
            StorageError: If storage initialization fails
        """
        try:
            # Create config with any overrides
            if kwargs:
                self.config = create_config(**kwargs)
            else:
                self.config = config
            
            # Initialize storage with config
            storage_dir = self.config.DEFAULT_STORAGE_DIR
            
            # Create storage instances
            self.prompt_storage = LocalStorage(model_class=Prompt, storage_dir=storage_dir)
            self.instance_storage = LocalStorage(model_class=PromptInstance, storage_dir=storage_dir)
            self.response_storage = LocalStorage(model_class=Response, storage_dir=storage_dir)
            self.feedback_storage = LocalStorage(model_class=Feedback, storage_dir=storage_dir)

            # Create core components with config
            self.prompt_manager = PromptManager(self.prompt_storage)
            self.response_tracker = ResponseTracker(
                self.prompt_storage,
                self.instance_storage,
                self.response_storage
            )
            
            # Initialize feedback collector
            self.feedback_collector = FeedbackCollector(
                self.feedback_storage,
                self.response_storage,
                self.instance_storage
            )
            
            # Initialize optimizer engine with config values
            self.optimizer = OptimizerEngine(
                self.prompt_manager,
                self.feedback_collector,
                config_instance=self.config
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize PromptOptimizer: {str(e)}")
            raise StorageError(f"Failed to initialize storage: {str(e)}") from e

    def register_prompt(self, text: str, description: str = "") -> str:
        """Register a new prompt.
        
        Args:
            text: The prompt template text
            description: Optional description
            
        Returns:
            The prompt ID
            
        Raises:
            ValidationError: If text is empty
            StorageError: If saving fails
        """
        try:
            # Simple validation
            validate_not_empty(text, "Prompt text")
            
            prompt = self.prompt_manager.create_prompt(text=text, description=description)
            logger.info(f"Created prompt {prompt.id}")
            return prompt.id
            
        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            logger.error(f"Failed to register prompt: {str(e)}")
            raise StorageError(f"Failed to save prompt: {str(e)}") from e

    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID.
        
        Args:
            prompt_id: The prompt ID
            
        Returns:
            Prompt data as dictionary, None if not found
            
        Raises:
            ValidationError: If prompt_id is invalid
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            prompt = self.prompt_manager.get_prompt(prompt_id)
            
            if not prompt:
                return None
                
            return prompt.model_dump()
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_id}: {str(e)}")
            raise StorageError(f"Failed to retrieve prompt: {str(e)}") from e

    def validate_prompt_id(self, prompt_id: str) -> bool:
        """Check if a prompt ID exists.
        
        Args:
            prompt_id: The prompt ID to check
            
        Returns:
            True if prompt exists, False otherwise
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            return self.prompt_manager.get_prompt(prompt_id) is not None
        except ValidationError:
            return False
        except Exception as e:
            logger.warning(f"Error validating prompt ID {prompt_id}: {str(e)}")
            return False

    def record_prompt_use(self, prompt_id: str, formatted_text: str, 
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Record the use of a prompt.
        
        Args:
            prompt_id: ID of the prompt being used
            formatted_text: The formatted prompt text
            context: Optional context data
            
        Returns:
            The prompt instance ID
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            ValidationError: If inputs are invalid
            StorageError: If saving fails
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            validate_not_empty(formatted_text, "Formatted text")
            
            # Check if prompt exists
            if not self.validate_prompt_id(prompt_id):
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
            
            instance = self.response_tracker.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=formatted_text,
                context=context
            )
            
            logger.info(f"Recorded prompt use for {prompt_id}")
            return instance.id
            
        except (ValidationError, PromptNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to record prompt use: {str(e)}")
            raise StorageError(f"Failed to record prompt use: {str(e)}") from e

    def record_response(self, prompt_instance_id: str, content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a response to a prompt instance.
        
        Args:
            prompt_instance_id: ID of the prompt instance
            content: The response content
            metadata: Optional metadata
            
        Returns:
            The response ID
            
        Raises:
            ValidationError: If inputs are invalid
            StorageError: If saving fails
        """
        try:
            validate_not_empty(prompt_instance_id, "Prompt instance ID")
            validate_not_empty(content, "Response content")
            
            response = self.response_tracker.record_response(
                prompt_instance_id=prompt_instance_id,
                content=content,
                metadata=metadata
            )
            
            logger.info(f"Recorded response {response.id}")
            return response.id
            
        except ValidationError:
            raise
        except ValueError as e:
            # Convert ValueError from response_tracker to our exception
            raise ResponseNotFoundError(str(e)) from e
        except Exception as e:
            logger.error(f"Failed to record response: {str(e)}")
            raise StorageError(f"Failed to record response: {str(e)}") from e

    def record_feedback(self, response_id: str, is_positive: bool,
                       score: Optional[float] = None, comments: Optional[str] = None) -> str:
        """Record feedback for a response.
        
        Args:
            response_id: ID of the response
            is_positive: Whether feedback is positive
            score: Optional score (0-1)
            comments: Optional comments
            
        Returns:
            The feedback ID
            
        Raises:
            ResponseNotFoundError: If response doesn't exist
            ValidationError: If inputs are invalid
            StorageError: If saving fails
        """
        try:
            validate_not_empty(response_id, "Response ID")
            
            # Validate score if provided
            if score is not None:
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise ValidationError("Score must be a number between 0 and 1")
            
            feedback = self.feedback_collector.record_feedback(
                response_id=response_id,
                is_positive=is_positive,
                score=score,
                comments=comments
            )
            
            logger.info(f"Recorded feedback {feedback.id}")
            return feedback.id
            
        except ValidationError:
            raise
        except ValueError as e:
            # Convert ValueError from feedback_collector to our exception
            raise ResponseNotFoundError(str(e)) from e
        except Exception as e:
            logger.error(f"Failed to record feedback: {str(e)}")
            raise StorageError(f"Failed to record feedback: {str(e)}") from e

    def optimize_prompt(self, prompt_id: str, force: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Optimize a prompt based on feedback.
        
        Args:
            prompt_id: ID of the prompt to optimize
            force: Force optimization even if not ready
            
        Returns:
            New prompt ID if auto_apply=True, optimized text otherwise, None if not optimized
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            OptimizationError: If optimization fails
            ValidationError: If prompt_id is invalid
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Check if prompt exists
            if not self.validate_prompt_id(prompt_id):
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
            
            # Check readiness
            readiness = self.optimizer.check_optimization_readiness(prompt_id)
            
            if force or readiness["is_ready"]:
                result = self.optimizer.generate_optimization(prompt_id, force=force)
                
                if result and self.optimizer.auto_apply:
                    # If auto_apply, result should be the new prompt ID
                    logger.info(f"Auto-applied optimization for {prompt_id} -> {result}")
                    return result
                elif result:
                    # If not auto_apply, result should be the optimized text
                    logger.info(f"Generated optimization for {prompt_id}")
                    return result
                else:
                    logger.info(f"No optimization generated for {prompt_id}")
                    return None
            else:
                logger.info(f"Prompt {prompt_id} not ready for optimization")
                return None
                
        except (ValidationError, PromptNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to optimize prompt {prompt_id}: {str(e)}")
            raise OptimizationError(f"Optimization failed: {str(e)}") from e

    def get_optimization_stats(self, prompt_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Dictionary with optimization statistics
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            ValidationError: If prompt_id is invalid
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            if not self.validate_prompt_id(prompt_id):
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
            
            return self.optimizer.check_optimization_readiness(prompt_id)
            
        except (ValidationError, PromptNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to get optimization stats for {prompt_id}: {str(e)}")
            raise StorageError(f"Failed to get stats: {str(e)}") from e
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information.
        
        Returns:
            Dictionary with current configuration
        """
        try:
            return {
                'optimization_threshold': self.config.OPTIMIZATION_THRESHOLD,
                'strategy': self.config.DEFAULT_STRATEGY, 
                'model': self.config.LLM_MODEL,
                'storage_dir': self.config.DEFAULT_STORAGE_DIR,
                'auto_apply': self.config.AUTO_APPLY,
                'max_tokens': self.config.MAX_TOKENS,
                'temperature': self.config.TEMPERATURE
            }
        except Exception as e:
            logger.warning(f"Failed to get config info: {str(e)}")
            return {'error': 'Failed to get configuration info'}