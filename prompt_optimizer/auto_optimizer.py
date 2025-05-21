# prompt_optimizer/auto_optimizer.py

from typing import List, Dict, Any, Optional, Callable
import threading
import time
import logging
import os
from datetime import datetime, timedelta

from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.models import Prompt
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy

class AutoOptimizer:
    """Automatic optimization manager for multiple prompts/agents."""
    
    def __init__(self, 
                 config: Optional[OptimizerConfig] = None,
                 llm_service: Optional[LLMService] = None):
        """Initialize the auto optimizer.
        
        Args:
            config: Optimizer configuration
            llm_service: LLM service for generation
        """
        from prompt_optimizer.api import PromptOptimizer
        
        # Use default config if none provided
        self.config = config or OptimizerConfig()
        
        # Create API instance
        self.api = PromptOptimizer(storage_dir=self.config.storage_dir)
        
        # Create LLM service
        self.llm_service = llm_service or LLMService()
        
        # Create strategy
        self.strategy = self.config.create_strategy(self.llm_service)
        
        self.check_interval = self.config.check_interval_hours * 3600  # Convert to seconds
        self.monitored_prompts = {}  # prompt_id -> last_check_time
        self.running = False
        self.optimization_thread = None
        
        # Setup logging
        self.logger = logging.getLogger("auto_optimizer")
        if not self.logger.handlers:
            os.makedirs(os.path.join(self.config.storage_dir, "logs"), exist_ok=True)
            handler = logging.FileHandler(
                os.path.join(self.config.storage_dir, "logs", "auto_optimizer.log")
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.logger.info(f"Initialized AutoOptimizer with strategy: {self.strategy.name}")
    
    # [Rest of the AutoOptimizer class implementation remains similar]
    # The key difference is that now the _optimize_prompt method will use:
    
    def _optimize_prompt(self, prompt_id: str) -> Optional[str]:
        """Run optimization for a single prompt."""
        self.logger.info(f"Checking prompt {prompt_id} for optimization")
        
        # Get the current prompt
        prompt = self.api.prompt_manager.get_prompt(prompt_id)
        if not prompt:
            self.logger.error(f"Prompt {prompt_id} not found")
            return None
        
        # Get feedback for this prompt
        feedback_data = self._get_all_feedback_for_prompt(prompt_id)
        
        # Check if we have enough feedback using the strategy
        readiness = self.strategy.is_ready_for_optimization(feedback_data)
        if not readiness.get("ready", False):
            self.logger.info(f"Not ready for optimization: {readiness.get('reason', 'Unknown reason')}")
            return None
            
        # Extract user queries
        user_queries = [self._extract_query_from_feedback(item) for item in feedback_data]
        user_queries = [q for q in user_queries if q]  # Remove None values
        
        # Run optimization using the strategy
        result = self.strategy.optimize(
            production_prompt=prompt,
            feedback_data=feedback_data,
            user_queries=user_queries
        )
        
        if result.get("status") == "optimized" and result.get("new_prompt"):
            # Create a new version with the optimized prompt
            new_prompt = self.api.prompt_manager.update_prompt(
                prompt_id=prompt_id,
                text=result["new_prompt"],
                description=f"Optimized using {self.strategy.name} strategy"
            )
            
            self.logger.info(f"Optimized prompt {prompt_id} using {self.strategy.name}")
            return new_prompt.id
        else:
            self.logger.info(f"No optimization: {result.get('reason', 'Unknown reason')}")
            return None
        
    def start_automatic_optimization(self):
        """Start the automatic optimization process in a background thread."""
        if self.running:
            self.logger.warning("Automatic optimization is already running")
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True  # Make thread exit when main program exits
        )
        self.optimization_thread.start()
        self.logger.info("Started automatic optimization")
        
    def stop_automatic_optimization(self):
        """Stop the automatic optimization process."""
        if not self.running:
            self.logger.warning("Automatic optimization is not running")
            return
        
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=2.0)  # Wait for thread to finish
        self.logger.info("Stopped automatic optimization")
        
    def _optimization_loop(self):
        """Main loop for automatic optimization."""
        self.logger.info("Optimization loop started")
        
        while self.running:
            # Check each monitored prompt
            for prompt_id, last_check_time in list(self.monitored_prompts.items()):
                # Check if it's time to optimize this prompt
                current_time = datetime.now()
                if last_check_time is None or (current_time - last_check_time).total_seconds() >= self.check_interval:
                    # Update last check time
                    self.monitored_prompts[prompt_id] = current_time
                    
                    # Get feedback data and check readiness
                    feedback_data = self._get_all_feedback_for_prompt(prompt_id)
                    readiness = self.strategy.is_ready_for_optimization(feedback_data)
                    
                    # Log readiness assessment (this would show in your app's optimization activity)
                    from prompt_optimizer.api import PromptOptimizer
                    self.logger.info(f"Checked prompt {prompt_id}: Ready={readiness.get('ready', False)}, Feedback count={len(feedback_data)}, Reason={readiness.get('reason', 'Unknown')}")
                    
                    # Try to optimize the prompt
                    if readiness.get("ready", False):
                        try:
                            optimized_id = self._optimize_prompt(prompt_id)
                            if optimized_id:
                                self.logger.info(f"Automatically optimized prompt {prompt_id} -> {optimized_id}")
                        except Exception as e:
                            self.logger.error(f"Error optimizing prompt {prompt_id}: {str(e)}")
            
            # Sleep for a while before checking again
            # Use small chunks so we can exit quickly if needed
            for _ in range(min(30, int(self.check_interval / 10))):
                if not self.running:
                    break
                time.sleep(10)

    def _get_all_feedback_for_prompt(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a prompt."""
        return self.api.feedback_collector.get_feedback_for_prompt(prompt_id, include_responses=True)

    def _extract_query_from_feedback(self, feedback_item: Dict[str, Any]) -> Optional[str]:
        """Extract the user query from feedback data."""
        # Try to extract from formatted text if available
        if "formatted_prompt" in feedback_item:
            return feedback_item["formatted_prompt"]
        
        # Try to extract from prompt instance
        if "prompt_instance" in feedback_item and "formatted_text" in feedback_item["prompt_instance"]:
            return feedback_item["prompt_instance"]["formatted_text"]
        
        return None