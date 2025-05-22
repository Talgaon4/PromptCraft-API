# prompt_optimizer/core/optimizer_engine.py

"""Engine for optimizing prompts based on feedback."""

import logging
from typing import Dict, List, Optional, Any, Union

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.feedback_collector import FeedbackCollector
from prompt_optimizer.config import config, create_config
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.exceptions import (
    OptimizationError, PromptNotFoundError, LLMError, StorageError,
    validate_prompt_id, validate_not_empty
)

logger = logging.getLogger(__name__)


class OptimizerEngine:
    """Engine for optimizing prompts based on feedback."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        feedback_collector: FeedbackCollector,
        config_instance=None,
        **overrides
    ):
        """Initialize the optimizer engine.
        
        Args:
            prompt_manager: Manager for prompt operations
            feedback_collector: Collector for feedback data
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides
            
        Raises:
            OptimizationError: If initialization fails
        """
        try:
            self.prompt_manager = prompt_manager
            self.feedback_collector = feedback_collector
            
            # Handle configuration
            if overrides:
                self.config = create_config(**overrides)
            else:
                self.config = config_instance or config
            
            # Extract settings from config
            self.optimization_threshold = self.config.OPTIMIZATION_THRESHOLD
            self.auto_apply = self.config.AUTO_APPLY
            self.strategy_name = self.config.DEFAULT_STRATEGY
            self.confidence_level = self.config.CONFIDENCE_LEVEL
            
            # Create LLM service with same config
            self.llm_service = LLMService(config_instance=self.config)
            
            # Import and create strategy based on config
            self.strategy = self._create_strategy()
            
            logger.info(f"OptimizerEngine initialized with strategy: {self.strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OptimizerEngine: {str(e)}")
            raise OptimizationError(f"Engine initialization failed: {str(e)}") from e

    def _create_strategy(self):
        """Create the optimization strategy based on configuration.
        
        Returns:
            Strategy instance
            
        Raises:
            OptimizationError: If strategy creation fails
        """
        try:
            from prompt_optimizer.strategies.simple_ai_strategy import SimpleAIStrategy
            from prompt_optimizer.strategies.reward_model_bandit import RewardModelBanditStrategy
            
            if self.strategy_name == "simple_ai":
                return SimpleAIStrategy(
                    llm_service=self.llm_service,
                    config_instance=self.config
                )
            elif self.strategy_name == "reward_model_bandit":
                return RewardModelBanditStrategy(
                    llm_service=self.llm_service,
                    config_instance=self.config
                )
            else:
                # Default to simple AI with warning
                logger.warning(f"Unknown strategy '{self.strategy_name}', using 'simple_ai'")
                return SimpleAIStrategy(
                    llm_service=self.llm_service,
                    config_instance=self.config
                )
                
        except ImportError as e:
            logger.error(f"Failed to import strategy {self.strategy_name}: {str(e)}")
            raise OptimizationError(f"Strategy '{self.strategy_name}' not available: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to create strategy {self.strategy_name}: {str(e)}")
            raise OptimizationError(f"Strategy creation failed: {str(e)}") from e

    def check_optimization_readiness(self, prompt_id: str) -> Dict[str, Any]:
        """Check if a prompt is ready for optimization.
        
        Args:
            prompt_id: ID of the prompt to check
            
        Returns:
            Dictionary with readiness information
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            OptimizationError: If readiness check fails
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Verify prompt exists
            if not self.prompt_manager.get_prompt(prompt_id):
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
            
            # Get all feedback for this prompt
            feedback_data = self.feedback_collector.get_feedback_for_prompt(prompt_id, include_responses=True)
            
            # Use the strategy to determine readiness
            strategy_readiness = self.strategy.is_ready_for_optimization(feedback_data)
            
            # Also check against our threshold
            threshold_readiness = len(feedback_data) >= self.optimization_threshold
            
            return {
                "prompt_id": prompt_id,
                "is_ready": threshold_readiness and strategy_readiness.get("ready", False),
                "feedback_count": len(feedback_data),
                "threshold": self.optimization_threshold,
                "strategy_name": self.strategy.name,
                "strategy_assessment": strategy_readiness,
                "config_info": {
                    "auto_apply": self.auto_apply,
                    "confidence_level": self.confidence_level
                }
            }
            
        except PromptNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to check optimization readiness for {prompt_id}: {str(e)}")
            raise OptimizationError(f"Readiness check failed: {str(e)}") from e

    def generate_optimization(self, prompt_id: str, force: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Generate an optimization for a prompt.
        
        Args:
            prompt_id: ID of the prompt to optimize
            force: If True, ignore readiness checks
            
        Returns:
            If auto_apply is True, returns the new prompt ID.
            Otherwise, returns the optimized prompt text.
            Returns None if optimization wasn't possible.
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            OptimizationError: If optimization fails
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Check if we have enough feedback (unless forced)
            if not force:
                readiness = self.check_optimization_readiness(prompt_id)
                if not readiness["is_ready"]:
                    logger.info(f"Prompt {prompt_id} not ready for optimization")
                    return None
                
            # Get the current prompt
            current_prompt = self.prompt_manager.get_prompt(prompt_id)
            if not current_prompt:
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
                
            # Get all feedback for this prompt
            feedback_data = self.feedback_collector.get_feedback_for_prompt(prompt_id, include_responses=True)
            
            if not feedback_data and not force:
                logger.info(f"No feedback data for prompt {prompt_id}")
                return None
            
            # Extract user queries from feedback data
            user_queries = []
            for item in feedback_data:
                if 'formatted_prompt' in item:
                    user_queries.append(item['formatted_prompt'])
                elif 'prompt_instance' in item and 'formatted_text' in item['prompt_instance']:
                    user_queries.append(item['prompt_instance']['formatted_text'])
            
            # Use the strategy to optimize the prompt
            optimization_result = self.strategy.optimize(
                production_prompt=current_prompt,
                feedback_data=feedback_data,
                user_queries=user_queries
            )
            
            # Check if optimization was successful
            if optimization_result.get("status") == "optimized" and optimization_result.get("new_prompt"):
                optimized_text = optimization_result["new_prompt"]
                
                # Apply or return the optimization
                if self.auto_apply:
                    new_prompt_id = self.apply_optimization(prompt_id, optimized_text)
                    logger.info(f"Applied optimization for {prompt_id} -> {new_prompt_id}")
                    return new_prompt_id
                else:
                    logger.info(f"Generated optimization for {prompt_id}")
                    return optimized_text
            else:
                logger.info(f"No optimization generated for {prompt_id}: {optimization_result.get('reason', 'Unknown reason')}")
                return None
                
        except (PromptNotFoundError, OptimizationError):
            raise
        except LLMError as e:
            logger.error(f"LLM error during optimization: {str(e)}")
            raise OptimizationError(f"LLM service failed during optimization: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during optimization: {str(e)}")
            raise OptimizationError(f"Optimization failed: {str(e)}") from e

    def apply_optimization(self, prompt_id: str, optimized_text: str) -> str:
        """Apply an optimization to a prompt, creating a new version.
        
        Args:
            prompt_id: ID of the prompt to optimize
            optimized_text: The optimized prompt text
            
        Returns:
            ID of the newly created prompt version
            
        Raises:
            PromptNotFoundError: If original prompt doesn't exist
            OptimizationError: If optimization application fails
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            validate_not_empty(optimized_text, "Optimized text")
            
            # Get the original prompt for description
            original_prompt = self.prompt_manager.get_prompt(prompt_id)
            if not original_prompt:
                raise PromptNotFoundError(f"Original prompt {prompt_id} not found")
            
            description = f"Optimized from {prompt_id} using {self.strategy.name} strategy"
            if original_prompt.description:
                description += f" - {original_prompt.description}"
            
            # Update the prompt to create a new version
            updated_prompt = self.prompt_manager.update_prompt(
                prompt_id=prompt_id,
                text=optimized_text,
                description=description
            )
            
            logger.info(f"Applied optimization: {prompt_id} -> {updated_prompt.id}")
            return updated_prompt.id
            
        except PromptNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to apply optimization: {str(e)}")
            raise OptimizationError(f"Failed to apply optimization: {str(e)}") from e

    def get_optimization_history(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get the optimization history for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of optimization events
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            StorageError: If history retrieval fails
        """
        try:
            prompt_id = validate_prompt_id(prompt_id)
            
            # Verify prompt exists
            if not self.prompt_manager.get_prompt(prompt_id):
                raise PromptNotFoundError(f"Prompt {prompt_id} not found")
            
            # Get the prompt history
            history = self.prompt_manager.get_prompt_history(prompt_id)
            
            # Format the history
            result = []
            for prompt in history:
                # Skip the first version - it's not an optimization
                if prompt.version == 1:
                    continue
                    
                result.append({
                    "prompt_id": prompt.id,
                    "parent_id": prompt.parent_id,
                    "version": prompt.version,
                    "text": prompt.text,
                    "description": prompt.description,
                    "created_at": prompt.created_at
                })
                
            return result
            
        except PromptNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get optimization history for {prompt_id}: {str(e)}")
            raise StorageError(f"Failed to get optimization history: {str(e)}") from e

    def update_settings(self, **new_settings):
        """Update optimizer settings at runtime.
        
        Args:
            **new_settings: Settings to update
            
        Raises:
            OptimizationError: If settings update fails
        """
        try:
            updated = []
            
            if 'optimization_threshold' in new_settings:
                threshold = new_settings['optimization_threshold']
                if not isinstance(threshold, int) or threshold < 1:
                    raise ValueError("Optimization threshold must be a positive integer")
                self.optimization_threshold = threshold
                updated.append('optimization_threshold')
                
            if 'auto_apply' in new_settings:
                self.auto_apply = bool(new_settings['auto_apply'])
                updated.append('auto_apply')
                
            if 'strategy_name' in new_settings and new_settings['strategy_name'] != self.strategy_name:
                self.strategy_name = new_settings['strategy_name']
                self.strategy = self._create_strategy()  # Recreate strategy
                updated.append('strategy_name')
            
            if updated:
                logger.info(f"Optimizer settings updated: {updated}")
                
        except ValueError as e:
            raise OptimizationError(f"Invalid setting value: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to update optimizer settings: {str(e)}")
            raise OptimizationError(f"Settings update failed: {str(e)}") from e
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current optimizer settings.
        
        Returns:
            Dictionary with current settings
        """
        try:
            return {
                'optimization_threshold': self.optimization_threshold,
                'auto_apply': self.auto_apply,
                'strategy_name': self.strategy_name,
                'confidence_level': self.confidence_level,
                'llm_settings': self.llm_service.get_settings()
            }
        except Exception as e:
            logger.warning(f"Failed to get optimizer settings: {str(e)}")
            return {'error': 'Failed to get settings'}