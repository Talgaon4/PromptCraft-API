# prompt_optimizer/strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from prompt_optimizer.core.models import Prompt

class OptimizationStrategy(ABC):
    """Base interface for prompt optimization strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the strategy."""
        pass
    
    @abstractmethod
    def optimize(self, 
               production_prompt: Prompt,
               feedback_data: List[Dict[str, Any]],
               user_queries: List[str],
               **kwargs) -> Dict[str, Any]:
        """Run the optimization process.
        
        Args:
            production_prompt: Current production prompt
            feedback_data: Historical feedback data
            user_queries: Historical user queries
            kwargs: Strategy-specific parameters
            
        Returns:
            Optimization results including new prompt if successful
        """
        pass
    
    @abstractmethod
    def is_ready_for_optimization(self,
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if there's enough data to run optimization.
        
        Args:
            feedback_data: Historical feedback data
            
        Returns:
            Dict with ready status and relevant metrics
        """
        pass