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
from prompt_optimizer.response_objects import OperationResult, PromptResult, OptimizationResult, ValidationResult
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

    def register_prompt(self, text: str, description: str = "") -> PromptResult:
        """Register a new prompt.
        
        Args:
            text: The prompt template text
            description: Optional description
            
        Returns:
            PromptResult with the created prompt information
        """
        try:
            # Simple validation
            validate_not_empty(text, "Prompt text")
            
            prompt = self.prompt_manager.create_prompt(text=text, description=description)
            logger.info(f"Created prompt {prompt.id}")
            
            return PromptResult.success(
                prompt_data=prompt.model_dump(),
                message=f"Prompt created successfully with ID: {prompt.id}"
            )
            
        except ValidationError as e:
            logger.warning(f"Validation error creating prompt: {str(e)}")
            return PromptResult.failure(
                message="Invalid prompt data",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to register prompt: {str(e)}")
            return PromptResult.failure(
                message="Failed to create prompt",
                errors=[f"Storage error: {str(e)}"]
            )

    def get_prompt(self, prompt_id: str) -> PromptResult:
        """Get a prompt by ID.
        
        Args:
            prompt_id: The prompt ID
            
        Returns:
            PromptResult with the prompt data or failure information
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            prompt = self.prompt_manager.get_prompt(prompt_id)
            
            if not prompt:
                return PromptResult.failure(
                    message=f"Prompt {prompt_id} not found"
                )
                
            return PromptResult.success(
                prompt_data=prompt.model_dump(),
                message="Prompt retrieved successfully"
            )
            
        except ValidationError as e:
            logger.warning(f"Invalid prompt ID {prompt_id}: {str(e)}")
            return PromptResult.failure(
                message="Invalid prompt ID",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_id}: {str(e)}")
            return PromptResult.failure(
                message="Failed to retrieve prompt",
                errors=[f"Storage error: {str(e)}"]
            )

    def validate_prompt_id(self, prompt_id: str) -> ValidationResult:
        """Check if a prompt ID exists.
        
        Args:
            prompt_id: The prompt ID to check
            
        Returns:
            ValidationResult with validation status
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            exists = self.prompt_manager.get_prompt(prompt_id) is not None
            
            if exists:
                return ValidationResult.valid(
                    message=f"Prompt {prompt_id} exists",
                    details={"prompt_id": prompt_id, "exists": True}
                )
            else:
                return ValidationResult.invalid(
                    message=f"Prompt {prompt_id} not found",
                    details={"prompt_id": prompt_id, "exists": False}
                )
                
        except ValidationError as e:
            return ValidationResult.invalid(
                message="Invalid prompt ID format",
                errors=[str(e)],
                details={"prompt_id": prompt_id, "exists": False}
            )
        except Exception as e:
            logger.warning(f"Error validating prompt ID {prompt_id}: {str(e)}")
            return ValidationResult.invalid(
                message="Validation error",
                errors=[str(e)],
                details={"prompt_id": prompt_id, "exists": False}
            )

    def record_prompt_use(self, prompt_id: str, formatted_text: str, 
                         context: Optional[Dict[str, Any]] = None) -> OperationResult:
        """Record the use of a prompt.
        
        Args:
            prompt_id: ID of the prompt being used
            formatted_text: The formatted prompt text
            context: Optional context data
            
        Returns:
            OperationResult with the prompt instance ID in data field
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            validate_not_empty(formatted_text, "Formatted text")
            
            # Check if prompt exists
            validation = self.validate_prompt_id(prompt_id)
            if not validation.is_valid:
                return OperationResult(
                    is_successful=False,
                    message=f"Prompt {prompt_id} not found",
                    errors=["Invalid prompt ID"]
                )
            
            instance = self.response_tracker.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=formatted_text,
                context=context
            )
            
            logger.info(f"Recorded prompt use for {prompt_id}")
            return OperationResult(
                is_successful=True,
                data={"instance_id": instance.id, "prompt_id": prompt_id},
                message="Prompt usage recorded successfully"
            )
            
        except ValidationError as e:
            logger.warning(f"Invalid data for prompt use: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Invalid request data",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to record prompt use: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Failed to record prompt use",
                errors=[f"Storage error: {str(e)}"]
            )

    def record_response(self, prompt_instance_id: str, content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> OperationResult:
        """Record a response to a prompt instance.
        
        Args:
            prompt_instance_id: ID of the prompt instance
            content: The response content
            metadata: Optional metadata
            
        Returns:
            OperationResult with the response ID in data field
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
            return OperationResult(
                is_successful=True,
                data={"response_id": response.id, "prompt_instance_id": prompt_instance_id},
                message="Response recorded successfully"
            )
            
        except ValidationError as e:
            logger.warning(f"Invalid response data: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Invalid request data",
                errors=[str(e)]
            )
        except ValueError as e:
            # Convert ValueError from response_tracker to our standard response
            logger.warning(f"Response recording failed: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Prompt instance not found",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to record response: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Failed to record response",
                errors=[f"Storage error: {str(e)}"]
            )

    def record_feedback(self, response_id: str, score: float) -> OperationResult:
        """Record feedback for a response.
        
        Args:
            response_id: ID of the response
            score: Numeric score (0-1)
            
        Returns:
            OperationResult with the feedback ID in data field
        """
        try:
            validate_not_empty(response_id, "Response ID")
            
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValidationError("Score must be a number between 0 and 1")
            
            feedback = self.feedback_collector.record_feedback(
                response_id=response_id,
                score=score
            )
            
            logger.info(f"Recorded feedback {feedback.id}")
            return OperationResult(
                is_successful=True,
                data={"feedback_id": feedback.id, "response_id": response_id},
                message="Feedback recorded successfully"
            )
            
        except ValidationError as e:
            logger.warning(f"Invalid feedback data: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Invalid request data",
                errors=[str(e)]
            )
        except ValueError as e:
            # Convert ValueError from feedback_collector to our standard response
            logger.warning(f"Feedback recording failed: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Response not found",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to record feedback: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Failed to record feedback",
                errors=[f"Storage error: {str(e)}"]
            )

    def optimize_prompt(self, prompt_id: str, force: bool = False) -> OptimizationResult:
        """Optimize a prompt based on feedback.
        
        Args:
            prompt_id: ID of the prompt to optimize
            force: Force optimization even if not ready
            
        Returns:
            OptimizationResult with optimization status and details
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Check if prompt exists
            validation = self.validate_prompt_id(prompt_id)
            if not validation.is_valid:
                return OptimizationResult.failure(
                    prompt_id=prompt_id,
                    message="Prompt not found",
                    errors=["Invalid prompt ID"]
                )
            
            # Check readiness
            readiness = self.optimizer.check_optimization_readiness(prompt_id)
            
            if force or readiness["is_ready"]:
                result = self.optimizer.generate_optimization(prompt_id, force=force)
                
                if result and self.optimizer.auto_apply:
                    # If auto_apply, result should be the new prompt ID
                    logger.info(f"Auto-applied optimization for {prompt_id} -> {result}")
                    return OptimizationResult.success(
                        original_id=prompt_id,
                        new_id=str(result),
                        applied=True,
                        reason="Optimization auto-applied based on feedback analysis",
                        message="Prompt optimized and applied successfully"
                    )
                elif result:
                    # If not auto_apply, result should be the optimized text
                    logger.info(f"Generated optimization for {prompt_id}")
                    return OptimizationResult(
                        is_successful=True,
                        data={"optimized_text": str(result)},
                        message="Optimization generated successfully (not applied)",
                        original_prompt_id=prompt_id,
                        optimization_applied=False,
                        improvement_reason="Generated based on feedback analysis"
                    )
                else:
                    logger.info(f"No optimization generated for {prompt_id}")
                    return OptimizationResult.not_ready(
                        prompt_id=prompt_id,
                        readiness_info=readiness,
                        message="No optimization needed at this time"
                    )
            else:
                logger.info(f"Prompt {prompt_id} not ready for optimization")
                return OptimizationResult.not_ready(
                    prompt_id=prompt_id,
                    readiness_info=readiness,
                    message="Not enough feedback for optimization"
                )
                
        except ValidationError as e:
            logger.warning(f"Invalid prompt ID for optimization: {str(e)}")
            return OptimizationResult.failure(
                prompt_id=prompt_id,
                message="Invalid prompt ID",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to optimize prompt {prompt_id}: {str(e)}")
            return OptimizationResult.failure(
                prompt_id=prompt_id,
                message="Optimization failed",
                errors=[f"Optimization error: {str(e)}"]
            )

    def get_optimization_stats(self, prompt_id: str) -> OptimizationResult:
        """Get optimization statistics for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            OptimizationResult with readiness information and statistics
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Check if prompt exists
            validation = self.validate_prompt_id(prompt_id)
            if not validation.is_valid:
                return OptimizationResult.failure(
                    prompt_id=prompt_id,
                    message="Prompt not found",
                    errors=["Invalid prompt ID"]
                )
            
            readiness_info = self.optimizer.check_optimization_readiness(prompt_id)
            
            return OptimizationResult(
                is_successful=True,
                data=readiness_info,
                message="Optimization statistics retrieved successfully",
                original_prompt_id=prompt_id,
                readiness_info=readiness_info
            )
            
        except ValidationError as e:
            logger.warning(f"Invalid prompt ID for stats: {str(e)}")
            return OptimizationResult.failure(
                prompt_id=prompt_id,
                message="Invalid prompt ID",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Failed to get optimization stats for {prompt_id}: {str(e)}")
            return OptimizationResult.failure(
                prompt_id=prompt_id,
                message="Failed to get optimization statistics",
                errors=[f"Stats error: {str(e)}"]
            )
    
    def get_config_info(self) -> OperationResult:
        """Get current configuration information.
        
        Returns:
            OperationResult with current configuration
        """
        try:
            config_data = {
                'optimization_threshold': self.config.OPTIMIZATION_THRESHOLD,
                'strategy': self.config.DEFAULT_STRATEGY, 
                'model': self.config.LLM_MODEL,
                'storage_dir': self.config.DEFAULT_STORAGE_DIR,
                'auto_apply': self.config.AUTO_APPLY,
                'max_tokens': self.config.MAX_TOKENS,
                'temperature': self.config.TEMPERATURE
            }
            
            return OperationResult(
                is_successful=True,
                data=config_data,
                message="Configuration retrieved successfully"
            )
            
        except Exception as e:
            logger.warning(f"Failed to get config info: {str(e)}")
            return OperationResult(
                is_successful=False,
                message="Failed to get configuration",
                errors=[str(e)]
            )