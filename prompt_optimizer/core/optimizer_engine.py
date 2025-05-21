# prompt_optimizer/core/optimizer_engine.py

"""Engine for optimizing prompts based on feedback."""

from typing import Dict, List, Optional, Any, Union

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.feedback_collector import FeedbackCollector
from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.services.llm_service import LLMService


class OptimizerEngine:
    """Engine for optimizing prompts based on feedback."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        feedback_collector: FeedbackCollector,
        optimization_threshold: int = 10,
        auto_apply: bool = False,
        strategy_name: str = "simple_ai"
    ):
        """Initialize the optimizer engine.
        
        Args:
            prompt_manager: Manager for prompt operations
            feedback_collector: Collector for feedback data
            optimization_threshold: Minimum feedback needed for optimization
            auto_apply: Whether to automatically apply optimizations
            strategy_name: Name of the strategy to use (from config)
        """
        self.prompt_manager = prompt_manager
        self.feedback_collector = feedback_collector
        self.optimization_threshold = optimization_threshold
        self.auto_apply = auto_apply
        
        # Create LLM service
        self.llm_service = LLMService()
        
        # Use the config to create the appropriate strategy
        self.config = OptimizerConfig(
            strategy=strategy_name,
            min_feedback_samples=optimization_threshold
        )
        self.strategy = self.config.create_strategy(self.llm_service)

    def check_optimization_readiness(self, prompt_id: str) -> Dict[str, Any]:
        """Check if a prompt is ready for optimization.
        
        Args:
            prompt_id: ID of the prompt to check
            
        Returns:
            Dictionary with readiness information
        """
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
            "strategy_assessment": strategy_readiness
        }

    def generate_optimization(self, prompt_id: str, force: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Generate an optimization for a prompt.
        
        Args:
            prompt_id: ID of the prompt to optimize
            force: If True, ignore readiness checks
            
        Returns:
            If auto_apply is True, returns the new prompt ID.
            Otherwise, returns the optimized prompt text.
            Returns None if optimization wasn't possible.
        """
        # Check if we have enough feedback (unless forced)
        if not force:
            readiness = self.check_optimization_readiness(prompt_id)
            if not readiness["is_ready"]:
                return None
            
        # Get the current prompt
        current_prompt = self.prompt_manager.get_prompt(prompt_id)
        if not current_prompt:
            raise ValueError(f"Prompt with ID {prompt_id} not found")
            
        # Get all feedback for this prompt
        feedback_data = self.feedback_collector.get_feedback_for_prompt(prompt_id, include_responses=True)
        
        # Extract user queries from feedback data
        user_queries = []
        for item in feedback_data:
            if 'formatted_prompt' in item:
                user_queries.append(item['formatted_prompt'])
            elif 'prompt_instance' in item and 'formatted_text' in item['prompt_instance']:
                user_queries.append(item['prompt_instance']['formatted_text'])
        
        # Use the strategy to optimize the prompt
        try:
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
                    return self.apply_optimization(prompt_id, optimized_text)
                else:
                    return optimized_text
            
            # Optimization wasn't successful
            return None
            
        except Exception as e:
            # Re-raise the exception
            raise

    def apply_optimization(self, prompt_id: str, optimized_text: str) -> str:
        """Apply an optimization to a prompt, creating a new version.
        
        Args:
            prompt_id: ID of the prompt to optimize
            optimized_text: The optimized prompt text
            
        Returns:
            ID of the newly created prompt version
        """
        # Get the original prompt for description
        original_prompt = self.prompt_manager.get_prompt(prompt_id)
        description = f"Optimized from {prompt_id} using {self.strategy.name} strategy"
        
        if original_prompt and original_prompt.description:
            description += f" - {original_prompt.description}"
        
        # Update the prompt to create a new version
        updated_prompt = self.prompt_manager.update_prompt(
            prompt_id=prompt_id,
            text=optimized_text,
            description=description
        )
        
        return updated_prompt.id

    def get_optimization_history(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get the optimization history for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of optimization events
        """
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

    def compare_versions(self, original_id: str, optimized_id: str) -> Dict[str, Any]:
        """Compare an original prompt with its optimized version.
        
        Args:
            original_id: ID of the original prompt
            optimized_id: ID of the optimized prompt
            
        Returns:
            Comparison information
        """
        original = self.prompt_manager.get_prompt(original_id)
        optimized = self.prompt_manager.get_prompt(optimized_id)
        
        if not original or not optimized:
            raise ValueError(f"One or both prompts not found: {original_id}, {optimized_id}")
            
        # Get feedback stats for both
        original_stats = self.feedback_collector.calculate_feedback_stats(original_id)
        optimized_stats = self.feedback_collector.calculate_feedback_stats(optimized_id)
        
        # Calculate improvement
        original_positive_rate = original_stats.get("positive_ratio", 0)
        optimized_positive_rate = optimized_stats.get("positive_ratio", 0)
        improvement = optimized_positive_rate - original_positive_rate
        
        return {
            "original": {
                "id": original_id,
                "text": original.text,
                "version": original.version,
                "stats": original_stats
            },
            "optimized": {
                "id": optimized_id,
                "text": optimized.text,
                "version": optimized.version,
                "stats": optimized_stats
            },
            "improvement": improvement,
            "improvement_percentage": f"{improvement * 100:.1f}%" if original_positive_rate > 0 else "N/A"
        }