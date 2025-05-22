"""Main API interface for the Prompt Optimizer."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from prompt_optimizer.core.models import Prompt, PromptInstance, Response, Feedback
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.response_tracker import ResponseTracker
from prompt_optimizer.core.feedback_collector import FeedbackCollector
from prompt_optimizer.core.optimizer_engine import OptimizerEngine
from prompt_optimizer.storage.local_storage import LocalStorage
from prompt_optimizer.config import config, create_config


class PromptOptimizer:
    """Main API interface for the Prompt Optimizer with flexible configuration."""

    def __init__(self, **kwargs):
        """Initialize the Prompt Optimizer API.
        
        Args:
            **kwargs: Configuration overrides. Common parameters:
                - optimization_threshold (int): Number of feedback items needed
                - strategy (str): Optimization strategy ('simple_ai', 'reward_model_bandit')
                - model (str): LLM model to use ('gpt-3.5-turbo', 'gpt-4')
                - storage_dir (str): Directory for data storage
                - max_tokens (int): Maximum tokens for LLM responses
                - temperature (float): LLM temperature setting
                - auto_apply (bool): Whether to auto-apply optimizations
                
        Examples:
            # Use defaults
            optimizer = PromptOptimizer()
            
            # Override specific parameters
            optimizer = PromptOptimizer(
                optimization_threshold=10,
                model="gpt-4",
                storage_dir="./my_data"
            )
            
            # Use different strategy
            optimizer = PromptOptimizer(
                strategy="reward_model_bandit",
                optimization_threshold=20
            )
        """
        
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
            optimization_threshold=self.config.OPTIMIZATION_THRESHOLD,
            auto_apply=self.config.AUTO_APPLY,
            strategy_name=self.config.DEFAULT_STRATEGY
        )

    def register_prompt(self, text: str, description: str = "") -> str:
        """Register a new prompt. Returns the prompt ID."""
        prompt = self.prompt_manager.create_prompt(text=text, description=description)
        return prompt.id

    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by ID."""
        prompt = self.prompt_manager.get_prompt(prompt_id)
        if not prompt:
            return None
        return prompt.model_dump()

    def validate_prompt_id(self, prompt_id: str) -> bool:
        """Check if a prompt ID exists."""
        return self.prompt_manager.get_prompt(prompt_id) is not None

    def record_prompt_use(self, prompt_id: str, formatted_text: str, 
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Record the use of a prompt. Returns the prompt instance ID."""
        instance = self.response_tracker.record_prompt_use(
            prompt_id=prompt_id,
            formatted_text=formatted_text,
            context=context
        )
        return instance.id

    def record_response(self, prompt_instance_id: str, content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a response to a prompt instance. Returns the response ID."""
        response = self.response_tracker.record_response(
            prompt_instance_id=prompt_instance_id,
            content=content,
            metadata=metadata
        )
        return response.id

    def record_feedback(self, response_id: str, is_positive: bool,
                       score: Optional[float] = None, comments: Optional[str] = None) -> str:
        """Record feedback for a response. Returns the feedback ID."""
        feedback = self.feedback_collector.record_feedback(
            response_id=response_id,
            is_positive=is_positive,
            score=score,
            comments=comments
        )
        return feedback.id

    def optimize_prompt(self, prompt_id: str, force: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Optimize a prompt based on feedback."""
        readiness = self.optimizer.check_optimization_readiness(prompt_id)
        
        if force or readiness["is_ready"]:
            if self.optimizer.auto_apply or force:
                optimized_text = self.optimizer.generate_optimization(prompt_id)
                if optimized_text:
                    return self.optimizer.apply_optimization(prompt_id, optimized_text)
                return None
            else:
                return self.optimizer.generate_optimization(prompt_id)
        
        return None

    def get_optimization_stats(self, prompt_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a prompt."""
        return self.optimizer.check_optimization_readiness(prompt_id)
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information (useful for debugging)."""
        return {
            'optimization_threshold': self.config.OPTIMIZATION_THRESHOLD,
            'strategy': self.config.DEFAULT_STRATEGY, 
            'model': self.config.LLM_MODEL,
            'storage_dir': self.config.DEFAULT_STORAGE_DIR,
            'auto_apply': self.config.AUTO_APPLY,
            'max_tokens': self.config.MAX_TOKENS,
            'temperature': self.config.TEMPERATURE
        }