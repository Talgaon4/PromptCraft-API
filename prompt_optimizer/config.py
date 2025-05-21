# prompt_optimizer/config.py

from typing import Dict, Any, Optional, List, Type
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy
from prompt_optimizer.strategies.reward_model_bandit import RewardModelBanditStrategy
from prompt_optimizer.strategies.simple_ai_strategy import SimpleAIStrategy

class OptimizerConfig:
    """Configuration for the prompt optimizer."""
    
    # Available strategies
    STRATEGIES = {
        "reward_model_bandit": RewardModelBanditStrategy,
        "simple_ai": SimpleAIStrategy
    }
    
    def __init__(self,
                 strategy: str = "simple_ai",
                 storage_dir: str = "./data",
                 min_feedback_samples: int = 5,
                 optimization_threshold: float = 0.05,
                 check_interval_hours: int = 1,
                 strategy_params: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            strategy: Name of the optimization strategy
            storage_dir: Directory to store data
            min_feedback_samples: Minimum feedback required
            optimization_threshold: Minimum improvement required
            check_interval_hours: How often to check for optimization
            strategy_params: Strategy-specific parameters
        """
        self.strategy_name = strategy
        self.storage_dir = storage_dir
        self.min_feedback_samples = min_feedback_samples
        self.optimization_threshold = optimization_threshold
        self.check_interval_hours = check_interval_hours
        self.strategy_params = strategy_params or {}
        
        # Validate strategy
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {', '.join(self.STRATEGIES.keys())}")
    
    def get_strategy_class(self) -> Type[OptimizationStrategy]:
        """Get the class for the selected strategy."""
        return self.STRATEGIES[self.strategy_name]
    
    def create_strategy(self, llm_service) -> OptimizationStrategy:
        """Create an instance of the selected strategy.
        
        Args:
            llm_service: LLM service for the strategy
            
        Returns:
            Instantiated strategy
        """
        strategy_class = self.get_strategy_class()
        
        # Different init params depending on strategy
        if self.strategy_name == "reward_model_bandit":
            return strategy_class(
                llm_service=llm_service,
                model_dir=f"{self.storage_dir}/models",
                optimization_threshold=self.optimization_threshold,
                min_feedback_samples=self.min_feedback_samples,
                **self.strategy_params
            )
        elif self.strategy_name == "simple_ai":
            return strategy_class(
                llm_service=llm_service,
                min_feedback_samples=self.min_feedback_samples,
                **self.strategy_params
            )
        else:
            # Generic fallback
            return strategy_class(llm_service=llm_service, **self.strategy_params)
    
    @classmethod
    def available_strategies(cls) -> List[Dict[str, str]]:
        """Get information about available strategies.
        
        Returns:
            List of dictionaries with strategy name and description
        """
        result = []
        for name, strategy_class in cls.STRATEGIES.items():
            # Create a temporary instance to get description
            try:
                description = strategy_class.description.fget(None)
            except:
                description = "No description available"
                
            result.append({
                "name": name,
                "description": description
            })
        return result