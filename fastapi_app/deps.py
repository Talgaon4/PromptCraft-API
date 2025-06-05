from prompt_optimizer.api import PromptOptimizer


def get_optimizer() -> PromptOptimizer:
    """Dependency that returns a PromptOptimizer instance."""
    return PromptOptimizer()
