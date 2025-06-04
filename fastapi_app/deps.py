from functools import lru_cache
from prompt_optimizer.api import PromptOptimizer

@lru_cache()
def get_optimizer() -> PromptOptimizer:
    return PromptOptimizer()
