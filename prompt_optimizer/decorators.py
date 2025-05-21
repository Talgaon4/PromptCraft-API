# prompt_optimizer/decorators.py

import functools
from typing import Any, Callable, Dict, Optional

from prompt_optimizer.auto_optimizer import AutoOptimizer

def optimize_prompt(prompt_text: str, 
                   description: str = "",
                   optimizer_instance: Optional[AutoOptimizer] = None):
    """Decorator that automatically optimizes a prompt-based function.
    
    Args:
        prompt_text: Initial prompt template
        description: Optional description
        optimizer_instance: Optional existing optimizer
        
    Returns:
        Decorated function
    """
    # Create or use optimizer
    optimizer = optimizer_instance or AutoOptimizer()
    
    # Register prompt
    prompt_id = optimizer.register_prompt(prompt_text, description)
    
    # Start optimization if not already running
    if not optimizer.running:
        optimizer.start_automatic_optimization()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the latest optimized prompt
            prompt_data = optimizer.api.get_prompt(prompt_id)
            prompt_template = prompt_data["text"]
            
            # Execute the function with the optimized prompt
            result, formatted_prompt, generated_response = func(prompt_template, *args, **kwargs)
            
            # Record the usage for optimization
            instance_id = optimizer.api.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=formatted_prompt
            )
            
            response_id = optimizer.api.record_response(
                prompt_instance_id=instance_id,
                content=generated_response
            )
            
            # Add response_id to result if it's a dict
            if isinstance(result, dict):
                result["response_id"] = response_id
            
            return result
        
        # Add a method to record feedback
        def record_feedback(response_id: str, is_positive: bool, 
                           score: Optional[float] = None, 
                           comments: Optional[str] = None):
            optimizer.api.record_feedback(
                response_id=response_id,
                is_positive=is_positive,
                score=score,
                comments=comments
            )
        
        wrapper.record_feedback = record_feedback
        
        return wrapper
    
    return decorator