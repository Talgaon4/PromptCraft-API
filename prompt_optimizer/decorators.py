# prompt_optimizer/decorators.py

import functools
import logging
from typing import Any, Callable, Dict, Optional

from prompt_optimizer.auto_optimizer import AutoOptimizer

# Setup logging
logger = logging.getLogger(__name__)

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
    
    # Register prompt using new API response format
    register_result = optimizer.api.register_prompt(prompt_text, description)
    
    if not register_result.success:
        logger.error(f"Failed to register prompt: {register_result.message}")
        logger.error(f"Errors: {register_result.errors}")
        raise RuntimeError(f"Prompt registration failed: {register_result.message}")
    
    prompt_id = register_result.prompt_id
    logger.info(f"Registered prompt with ID: {prompt_id}")
    
    # Add prompt to monitoring
    monitoring_added = optimizer.add_prompt_to_monitoring(prompt_id)
    if monitoring_added:
        logger.info(f"Added prompt {prompt_id} to monitoring")
    else:
        logger.warning(f"Failed to add prompt {prompt_id} to monitoring")
    
    # Start optimization if not already running
    if not optimizer.running:
        optimizer.start_automatic_optimization()
        logger.info("Started automatic optimization")
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Get the latest optimized prompt using new API response format
                prompt_result = optimizer.api.get_prompt(prompt_id)
                
                if not prompt_result.success:
                    logger.warning(f"Failed to get current prompt: {prompt_result.message}")
                    logger.warning(f"Using original prompt text as fallback")
                    prompt_template = prompt_text
                else:
                    prompt_template = prompt_result.prompt_text or prompt_text
                    logger.debug(f"Using prompt version {prompt_result.version}")
                
                # Execute the function with the optimized prompt
                result, formatted_prompt, generated_response = func(prompt_template, *args, **kwargs)
                
                # Record the usage for optimization using new API response format
                usage_result = optimizer.api.record_prompt_use(
                    prompt_id=prompt_id,
                    formatted_text=formatted_prompt
                )
                
                if not usage_result.success:
                    logger.error(f"Failed to record prompt use: {usage_result.message}")
                    logger.error(f"Errors: {usage_result.errors}")
                    # Continue execution but log the error
                    return result
                
                instance_id = usage_result.data.get('instance_id')
                logger.debug(f"Recorded prompt usage: {instance_id}")
                
                # Record the response using new API response format
                response_result = optimizer.api.record_response(
                    prompt_instance_id=instance_id,
                    content=generated_response
                )
                
                if not response_result.success:
                    logger.error(f"Failed to record response: {response_result.message}")
                    logger.error(f"Errors: {response_result.errors}")
                    # Continue execution but log the error
                    return result
                
                response_id = response_result.data.get('response_id')
                logger.debug(f"Recorded response: {response_id}")
                
                # Add response_id to result if it's a dict
                if isinstance(result, dict):
                    result["response_id"] = response_id
                    result["prompt_id"] = prompt_id
                    result["instance_id"] = instance_id
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimized function wrapper: {str(e)}")
                # Try to execute the function with original prompt as fallback
                try:
                    logger.info("Attempting fallback execution with original prompt")
                    fallback_result, _, _ = func(prompt_text, *args, **kwargs)
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback execution also failed: {str(fallback_error)}")
                    raise e  # Raise the original error
        
        # Add a method to record feedback using new API response format
        def record_feedback(response_id: str, score: float):
            """Record feedback for a response generated by the decorated function.
            
            Args:
                response_id: ID of the response to provide feedback for
                score: Feedback score between 0 and 1
                
            Returns:
                True if feedback was recorded successfully, False otherwise
            """
            try:
                feedback_result = optimizer.api.record_feedback(
                    response_id=response_id,
                    score=score
                )
                
                if feedback_result.success:
                    feedback_id = feedback_result.data.get('feedback_id')
                    logger.info(f"Recorded feedback: {feedback_id}")
                    logger.debug(f"Feedback score: {score}")
                    return True
                else:
                    logger.error(f"Failed to record feedback: {feedback_result.message}")
                    logger.error(f"Errors: {feedback_result.errors}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error recording feedback: {str(e)}")
                return False
        
        # Add a method to get optimization status
        def get_optimization_status():
            """Get the current optimization status for this prompt.
            
            Returns:
                Dictionary with optimization status information
            """
            try:
                stats_result = optimizer.api.get_optimization_stats(prompt_id)
                
                if stats_result.success:
                    return {
                        "success": True,
                        "prompt_id": prompt_id,
                        "readiness_info": stats_result.readiness_info,
                        "message": stats_result.message
                    }
                else:
                    return {
                        "success": False,
                        "prompt_id": prompt_id,
                        "message": stats_result.message,
                        "errors": stats_result.errors
                    }
                    
            except Exception as e:
                logger.error(f"Error getting optimization status: {str(e)}")
                return {
                    "success": False,
                    "prompt_id": prompt_id,
                    "message": f"Error getting status: {str(e)}"
                }
        
        # Add a method to manually trigger optimization
        def optimize_now(force: bool = False):
            """Manually trigger optimization for this prompt.
            
            Args:
                force: Whether to force optimization even if not ready
                
            Returns:
                Dictionary with optimization results
            """
            try:
                opt_result = optimizer.api.optimize_prompt(prompt_id, force=force)
                
                return {
                    "success": opt_result.success,
                    "message": opt_result.message,
                    "optimization_applied": opt_result.optimization_applied,
                    "new_prompt_id": opt_result.new_prompt_id,
                    "improvement_reason": opt_result.improvement_reason,
                    "readiness_info": opt_result.readiness_info,
                    "errors": opt_result.errors
                }
                
            except Exception as e:
                logger.error(f"Error during manual optimization: {str(e)}")
                return {
                    "success": False,
                    "message": f"Optimization error: {str(e)}"
                }
        
        # Attach utility methods to the wrapper function
        wrapper.record_feedback = record_feedback
        wrapper.get_optimization_status = get_optimization_status
        wrapper.optimize_now = optimize_now
        wrapper.prompt_id = prompt_id
        wrapper.optimizer = optimizer
        
        return wrapper
    
    return decorator


# Example usage and testing helper
def create_simple_optimizer_decorator(prompt_text: str, description: str = ""):
    """Create a simple version of the decorator for quick testing.
    
    Args:
        prompt_text: The prompt template
        description: Optional description
        
    Returns:
        Decorator function
    
    Example:
        @create_simple_optimizer_decorator("Summarize: {text}")
        def summarize(prompt, text):
            # Your LLM call here
            formatted = prompt.replace("{text}", text)
            response = "Summary of the text"  # Replace with actual LLM call
            return {"summary": response}, formatted, response
    """
    return optimize_prompt(prompt_text, description)


# Utility function for batch feedback recording
def record_batch_feedback(decorated_function, feedback_batch: list):
    """Record multiple feedback items for a decorated function.
    
    Args:
        decorated_function: Function decorated with @optimize_prompt
        feedback_batch: List of dicts with keys: response_id, score
        
    Returns:
        Dictionary with success count and any errors
    """
    if not hasattr(decorated_function, 'record_feedback'):
        raise ValueError("Function must be decorated with @optimize_prompt")
    
    results = {
        "total": len(feedback_batch),
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    for item in feedback_batch:
        try:
            success = decorated_function.record_feedback(
                response_id=item.get('response_id'),
                score=item.get('score', 0.0)
            )
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"Failed to record feedback for {item.get('response_id')}")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Error processing {item.get('response_id')}: {str(e)}")
    
    logger.info(f"Batch feedback recorded: {results['successful']}/{results['total']} successful")
    return results