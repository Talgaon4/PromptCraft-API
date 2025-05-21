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


class PromptOptimizer:
    """Main API interface for the Prompt Optimizer."""

    def __init__(
        self,
        storage_dir: str = "./data",
        optimization_threshold: int = 10,
        auto_apply: bool = False,
        strategy_name: str = "simple_ai"  # Added the strategy_name parameter
    ):
        """Initialize the Prompt Optimizer API."""
        # Create storage instances
        self.prompt_storage = LocalStorage(model_class=Prompt, storage_dir=storage_dir)
        self.instance_storage = LocalStorage(model_class=PromptInstance, storage_dir=storage_dir)
        self.response_storage = LocalStorage(model_class=Response, storage_dir=storage_dir)
        self.feedback_storage = LocalStorage(model_class=Feedback, storage_dir=storage_dir)

        # Create core components
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
            self.instance_storage  # Pass instance storage to FeedbackCollector
        )
        
        # Initialize optimizer engine with strategy name
        self.optimizer = OptimizerEngine(
            self.prompt_manager,
            self.feedback_collector,
            optimization_threshold=optimization_threshold,
            auto_apply=auto_apply,
            strategy_name=strategy_name  # Pass the strategy_name parameter
        )

    def register_prompt(
        self,
        text: str,
        description: str = ""
    ) -> str:
        """Register a new prompt.
        
        Returns the prompt ID.
        """
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

    def record_prompt_use(
        self,
        prompt_id: str,
        formatted_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record the use of a prompt.
        
        Returns the prompt instance ID.
        """
        instance = self.response_tracker.record_prompt_use(
            prompt_id=prompt_id,
            formatted_text=formatted_text,
            context=context
        )
        return instance.id

    def record_response(
        self,
        prompt_instance_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a response to a prompt instance.
        
        Returns the response ID.
        """
        response = self.response_tracker.record_response(
            prompt_instance_id=prompt_instance_id,
            content=content,
            metadata=metadata
        )
        return response.id

    def record_feedback(
        self,
        response_id: str,
        is_positive: bool,
        score: Optional[float] = None,
        comments: Optional[str] = None
    ) -> str:
        """Record feedback for a response.
        
        Returns the feedback ID.
        """
        feedback = self.feedback_collector.record_feedback(
            response_id=response_id,
            is_positive=is_positive,
            score=score,
            comments=comments
        )
        return feedback.id

    def optimize_prompt(self, prompt_id: str, force: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Optimize a prompt based on feedback.
        
        If auto_apply is True or force is True, applies the optimization and returns the new prompt ID.
        Otherwise, returns the optimized prompt text without applying it.
        
        Returns None if there's not enough feedback to optimize.
        """
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
