# prompt_optimizer/auto_optimizer.py

"""Automatic optimization manager for multiple prompts/agents."""

from typing import List, Dict, Any, Optional, Callable
import threading
import time
import logging
import os
from datetime import datetime, timedelta

from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.core.models import Prompt
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.config import config, create_config

class AutoOptimizer:
    """Automatic optimization manager for multiple prompts/agents."""
    
    def __init__(self, 
                 config_instance=None,
                 **overrides):
        """Initialize the auto optimizer.
        
        Args:
            config_instance: Custom config instance (optional)
            **overrides: Configuration overrides:
                - storage_dir: Directory for data storage
                - check_interval_hours: Hours between optimization checks
                - check_interval_seconds: Seconds between checks (overrides hours for demo mode)
                - strategy: Optimization strategy name
                - log_level: Logging level
        """
        from prompt_optimizer.api import PromptOptimizer
        
        # Handle configuration
        if overrides:
            self.config = create_config(**overrides)
        else:
            self.config = config_instance or config
        
        # Extract timing settings
        self.check_interval_hours = getattr(self.config, 'AUTO_CHECK_INTERVAL_HOURS', 1)
        self.check_interval_seconds = getattr(self.config, 'AUTO_CHECK_INTERVAL_SECONDS', 30)
        
        # Use seconds if specified, otherwise convert hours to seconds
        if hasattr(self.config, 'AUTO_CHECK_INTERVAL_SECONDS'):
            self.check_interval = self.check_interval_seconds
        else:
            self.check_interval = self.check_interval_hours * 3600
        
        # Create API instance with same config
        self.api = PromptOptimizer(config_instance=self.config)
        
        # Create LLM service
        self.llm_service = LLMService(config_instance=self.config)
        
        # Create strategy based on config
        self.strategy = self._create_strategy()
        
        # Monitoring state
        self.monitored_prompts = {}  # prompt_id -> last_check_time
        self.running = False
        self.optimization_thread = None
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Initialized AutoOptimizer with strategy: {self.strategy.name}")
        self.logger.info(f"Check interval: {self.check_interval} seconds")
    
    def _create_strategy(self):
        """Create optimization strategy based on configuration."""
        from prompt_optimizer.strategies.simple_ai_strategy import SimpleAIStrategy
        from prompt_optimizer.strategies.reward_model_bandit import RewardModelBanditStrategy
        
        strategy_name = self.config.DEFAULT_STRATEGY
        
        if strategy_name == "simple_ai":
            return SimpleAIStrategy(
                llm_service=self.llm_service,
                config_instance=self.config
            )
        elif strategy_name == "reward_model_bandit":
            return RewardModelBanditStrategy(
                llm_service=self.llm_service,
                config_instance=self.config
            )
        else:
            # Default to simple AI
            self.logger.warning(f"Unknown strategy '{strategy_name}', using 'simple_ai'")
            return SimpleAIStrategy(
                llm_service=self.llm_service,
                config_instance=self.config
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger("auto_optimizer")
        
        if not self.logger.handlers:
            # Create logs directory
            logs_dir = getattr(self.config, 'LOGS_DIR', './logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Setup file handler
            log_file = os.path.join(logs_dir, "auto_optimizer.log")
            handler = logging.FileHandler(log_file)
            
            # Setup formatter
            log_format = getattr(self.config, 'LOG_FORMAT', 
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)
            
            # Add handler and set level
            self.logger.addHandler(handler)
            log_level = getattr(self.config, 'LOG_LEVEL', 'INFO')
            self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def add_prompt_to_monitoring(self, prompt_id: str) -> bool:
        """Add a prompt to the monitoring list.
        
        Args:
            prompt_id: ID of the prompt to monitor
            
        Returns:
            True if added successfully, False if already monitored
        """
        if prompt_id in self.monitored_prompts:
            self.logger.warning(f"Prompt {prompt_id} is already being monitored")
            return False
        
        # Validate prompt exists using new API response format
        validation_result = self.api.validate_prompt_id(prompt_id)
        if not validation_result.success or not validation_result.is_valid:
            self.logger.error(f"Invalid prompt ID: {prompt_id}")
            self.logger.error(f"Validation failed: {validation_result.message}")
            return False
        
        self.monitored_prompts[prompt_id] = None  # Will be set on first check
        self.logger.info(f"Added prompt {prompt_id} to monitoring")
        return True
    
    def remove_prompt_from_monitoring(self, prompt_id: str) -> bool:
        """Remove a prompt from monitoring.
        
        Args:
            prompt_id: ID of the prompt to stop monitoring
            
        Returns:
            True if removed successfully, False if not found
        """
        if prompt_id not in self.monitored_prompts:
            self.logger.warning(f"Prompt {prompt_id} is not being monitored")
            return False
        
        del self.monitored_prompts[prompt_id]
        self.logger.info(f"Removed prompt {prompt_id} from monitoring")
        return True
    
    def start_automatic_optimization(self):
        """Start the automatic optimization process in a background thread."""
        if self.running:
            self.logger.warning("Automatic optimization is already running")
            return
        
        if not self.monitored_prompts:
            self.logger.warning("No prompts to monitor. Add prompts before starting.")
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
            try:
                # Check each monitored prompt
                for prompt_id, last_check_time in list(self.monitored_prompts.items()):
                    if not self.running:  # Exit if stopped
                        break
                    
                    # Check if it's time to optimize this prompt
                    current_time = datetime.now()
                    if last_check_time is None or (current_time - last_check_time).total_seconds() >= self.check_interval:
                        # Update last check time
                        self.monitored_prompts[prompt_id] = current_time
                        
                        # Try to optimize the prompt
                        self._check_and_optimize_prompt(prompt_id)
                
                # Sleep between cycles
                self._sleep_with_interrupt_check()
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                # Continue running even if there's an error
                time.sleep(10)  # Short pause before retrying
        
        self.logger.info("Optimization loop stopped")

    def _check_and_optimize_prompt(self, prompt_id: str):
        """Check if a prompt needs optimization and optimize if ready."""
        try:
            # Get feedback data and check readiness using new API response format
            feedback_data = self._get_all_feedback_for_prompt(prompt_id)
            readiness = self.strategy.is_ready_for_optimization(feedback_data)
            
            # Log readiness assessment
            self.logger.info(f"Checked prompt {prompt_id}: Ready={readiness.get('ready', False)}, "
                           f"Feedback count={len(feedback_data)}, "
                           f"Reason={readiness.get('reason', 'Unknown')}")
            
            # Try to optimize if ready
            if readiness.get("ready", False):
                optimization_result = self._optimize_prompt(prompt_id)
                if optimization_result:
                    self.logger.info(f"Successfully optimized prompt {prompt_id} -> {optimization_result}")
                else:
                    self.logger.info(f"Optimization not applied for prompt {prompt_id}")
            
        except Exception as e:
            self.logger.error(f"Error checking/optimizing prompt {prompt_id}: {str(e)}")

    def _optimize_prompt(self, prompt_id: str) -> Optional[str]:
        """Run optimization for a single prompt using new API response format."""
        try:
            # Use the API's optimization method with new response format
            result = self.api.optimize_prompt(prompt_id, force=False)
            
            if result.success:
                if result.optimization_applied and result.new_prompt_id:
                    # Optimization was applied, return new prompt ID
                    self.logger.info(f"Optimization applied: {prompt_id} -> {result.new_prompt_id}")
                    return result.new_prompt_id
                elif result.data and 'optimized_text' in result.data:
                    # Optimization generated but not applied
                    self.logger.info(f"Optimization generated for {prompt_id} (not applied)")
                    return result.data['optimized_text']
                else:
                    self.logger.info(f"Optimization completed but no changes made for {prompt_id}")
                    return None
            else:
                # Optimization failed or not ready
                self.logger.info(f"Optimization not performed for {prompt_id}: {result.message}")
                
                # Log readiness info if available
                if result.readiness_info:
                    readiness = result.readiness_info
                    self.logger.debug(f"Readiness info for {prompt_id}: "
                                    f"feedback_count={readiness.get('feedback_count', 0)}, "
                                    f"is_ready={readiness.get('is_ready', False)}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to optimize prompt {prompt_id}: {str(e)}")
            return None

    def _get_all_feedback_for_prompt(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a prompt."""
        try:
            return self.api.feedback_collector.get_feedback_for_prompt(prompt_id, include_responses=True)
        except Exception as e:
            self.logger.error(f"Failed to get feedback for prompt {prompt_id}: {str(e)}")
            return []

    def _sleep_with_interrupt_check(self):
        """Sleep with ability to interrupt quickly when stopping."""
        # Sleep in small chunks so we can exit quickly if needed
        sleep_chunk = min(10, self.check_interval // 10)  # Sleep in 10% chunks, max 10 seconds
        total_slept = 0
        
        while total_slept < self.check_interval and self.running:
            time.sleep(sleep_chunk)
            total_slept += sleep_chunk

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'running': self.running,
            'monitored_prompts': list(self.monitored_prompts.keys()),
            'check_interval': self.check_interval,
            'strategy': self.strategy.name,
            'config_info': {
                'storage_dir': self.config.DEFAULT_STORAGE_DIR,
                'strategy': self.config.DEFAULT_STRATEGY,
                'check_interval_hours': self.check_interval_hours,
                'check_interval_seconds': self.check_interval_seconds
            }
        }
    
    def update_settings(self, **new_settings):
        """Update auto-optimizer settings at runtime.
        
        Args:
            **new_settings: Settings to update
        """
        updated = []
        
        if 'check_interval' in new_settings:
            self.check_interval = new_settings['check_interval']
            updated.append('check_interval')
        
        if 'check_interval_hours' in new_settings:
            self.check_interval_hours = new_settings['check_interval_hours']
            self.check_interval = self.check_interval_hours * 3600
            updated.append('check_interval_hours')
            
        if 'check_interval_seconds' in new_settings:
            self.check_interval_seconds = new_settings['check_interval_seconds']
            self.check_interval = self.check_interval_seconds
            updated.append('check_interval_seconds')
        
        if 'strategy' in new_settings and new_settings['strategy'] != self.config.DEFAULT_STRATEGY:
            self.config.DEFAULT_STRATEGY = new_settings['strategy']
            self.strategy = self._create_strategy()
            updated.append('strategy')
        
        if updated:
            self.logger.info(f"AutoOptimizer settings updated: {updated}")
            
    def manual_check_all(self) -> Dict[str, Any]:
        """Manually check all monitored prompts for optimization opportunities.
        
        Returns:
            Dictionary with results for each prompt
        """
        results = {}
        
        for prompt_id in self.monitored_prompts.keys():
            try:
                # Update last check time
                self.monitored_prompts[prompt_id] = datetime.now()
                
                # Check and potentially optimize
                feedback_data = self._get_all_feedback_for_prompt(prompt_id)
                readiness = self.strategy.is_ready_for_optimization(feedback_data)
                
                result = {
                    'prompt_id': prompt_id,
                    'feedback_count': len(feedback_data),
                    'ready_for_optimization': readiness.get('ready', False),
                    'reason': readiness.get('reason', 'Unknown'),
                    'optimized': False,
                    'new_prompt_id': None
                }
                
                # Try to optimize if ready
                if readiness.get('ready', False):
                    optimized_result = self._optimize_prompt(prompt_id)
                    if optimized_result:
                        result['optimized'] = True
                        result['new_prompt_id'] = optimized_result
                
                results[prompt_id] = result
                
            except Exception as e:
                results[prompt_id] = {
                    'prompt_id': prompt_id,
                    'error': str(e),
                    'optimized': False
                }
        
        self.logger.info(f"Manual check completed for {len(results)} prompts")
        return results