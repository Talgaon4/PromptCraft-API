# prompt_optimizer/strategies/simple_ai_strategy.py

"""Simple prompt optimization using a language model."""

import logging
from typing import List, Dict, Any, Optional
from prompt_optimizer.core.models import Prompt
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.config import config
from prompt_optimizer.exceptions import (
    OptimizationError, LLMError, ValidationError, validate_not_empty
)

logger = logging.getLogger(__name__)


class SimpleAIStrategy(OptimizationStrategy):
    """Simple prompt optimization using a language model."""
    
    def __init__(self, 
                 llm_service: LLMService,
                 config_instance=None,
                 **overrides):
        """Initialize the simple AI strategy.
        
        Args:
            llm_service: Service for generating LLM responses
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides
            
        Raises:
            OptimizationError: If initialization fails
        """
        try:
            self.llm_service = llm_service
            
            # Handle configuration
            self.config = config_instance or config
            
            # Apply overrides or use config values
            self.min_feedback_samples = overrides.get('min_feedback_samples', 
                                                     self.config.SIMPLE_AI_MIN_SAMPLES)
            self.min_positive_rate = overrides.get('min_positive_rate', 
                                                  self.config.SIMPLE_AI_MIN_POSITIVE_RATE)
            
            # Validate configuration
            if self.min_feedback_samples < 1:
                raise ValueError("min_feedback_samples must be at least 1")
            if not 0 <= self.min_positive_rate <= 1:
                raise ValueError("min_positive_rate must be between 0 and 1")
                
            logger.info(f"SimpleAI strategy initialized with min_samples={self.min_feedback_samples}")
            
        except ValueError as e:
            raise OptimizationError(f"Invalid strategy configuration: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to initialize SimpleAI strategy: {str(e)}")
            raise OptimizationError(f"Strategy initialization failed: {str(e)}") from e
    
    @property
    def name(self) -> str:
        return "simple_ai"
    
    @property
    def description(self) -> str:
        return "Optimize prompts by asking an AI to analyze feedback and suggest improvements"
    
    def is_ready_for_optimization(self, 
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if there's enough feedback to optimize.
        
        Args:
            feedback_data: List of feedback items
            
        Returns:
            Dictionary with readiness information
        """
        try:
            if not isinstance(feedback_data, list):
                return {"ready": False, "reason": "Invalid feedback data format"}
                
            if not feedback_data:
                return {"ready": False, "reason": "No feedback available"}
            
            total_feedback = len(feedback_data)
            if total_feedback < self.min_feedback_samples:
                return {
                    "ready": False, 
                    "reason": f"Insufficient feedback: {total_feedback}/{self.min_feedback_samples}",
                    "current_samples": total_feedback,
                    "required_samples": self.min_feedback_samples
                }
            
            # Calculate positive rate
            positive_count = sum(1 for item in feedback_data if item.get("score", 0) >= 0.5)
            positive_rate = positive_count / total_feedback if total_feedback > 0 else 0
            
            # If positive rate is already very high, maybe no need to optimize
            if positive_rate >= 0.9:
                return {
                    "ready": False, 
                    "reason": f"Prompt is already performing excellently: {positive_rate:.1%} positive",
                    "positive_rate": positive_rate,
                    "current_samples": total_feedback
                }
            
            # If positive rate is too low, optimization is needed
            needs_optimization = positive_rate < self.min_positive_rate
            
            return {
                "ready": needs_optimization,
                "reason": f"Prompt needs improvement: {positive_rate:.1%} positive (target: {self.min_positive_rate:.1%})" if needs_optimization else f"Prompt is performing adequately: {positive_rate:.1%} positive",
                "positive_rate": positive_rate,
                "current_samples": total_feedback,
                "required_samples": self.min_feedback_samples,
                "target_positive_rate": self.min_positive_rate
            }
            
        except Exception as e:
            logger.error(f"Error checking optimization readiness: {str(e)}")
            return {"ready": False, "reason": f"Error checking readiness: {str(e)}"}
    
    def optimize(self, 
               production_prompt: Prompt,
               feedback_data: List[Dict[str, Any]],
               user_queries: List[str],
               **kwargs) -> Dict[str, Any]:
        """Optimize the prompt using AI.
        
        Args:
            production_prompt: Current production prompt
            feedback_data: Historical feedback data
            user_queries: Historical user queries
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Validate inputs
            if not production_prompt:
                raise ValidationError("Production prompt is required")
            if not isinstance(feedback_data, list):
                raise ValidationError("Feedback data must be a list")
            
            # Check if ready
            readiness = self.is_ready_for_optimization(feedback_data)
            if not readiness["ready"]:
                return {
                    "status": "not_optimized",
                    "reason": readiness["reason"],
                    "new_prompt": None,
                    "readiness_info": readiness
                }
            
            # Prepare examples for the AI (limit based on config or reasonable default)
            max_examples = getattr(self.config, 'SIMPLE_AI_MAX_EXAMPLES', 5)
            examples = self._prepare_examples(feedback_data, limit=max_examples)
            
            if not examples:
                return {
                    "status": "not_optimized",
                    "reason": "No usable feedback examples found",
                    "new_prompt": None
                }
            
            # Create the optimization prompt
            optimization_prompt = self._create_optimization_prompt(
                current_prompt=production_prompt.text,
                examples=examples
            )
            
            # Get optimization from AI
            improved_prompt = self.llm_service.optimize_prompt(
                current_prompt=production_prompt.text,
                feedback_examples=examples
            )
            
            # Clean up the response to extract just the prompt
            improved_prompt = self._extract_prompt(improved_prompt)
            
            # Validate the result
            if not improved_prompt or improved_prompt.strip() == production_prompt.text.strip():
                return {
                    "status": "not_optimized",
                    "reason": "AI did not generate a meaningful improvement",
                    "new_prompt": None
                }
            
            return {
                "status": "optimized",
                "reason": "AI suggestion applied",
                "original_prompt": production_prompt.text,
                "new_prompt": improved_prompt,
                "examples_analyzed": len(examples),
                "readiness_info": readiness
            }
            
        except (ValidationError, LLMError):
            raise
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Strategy optimization failed: {str(e)}") from e
    
    def _prepare_examples(self, 
                         feedback_data: List[Dict[str, Any]], 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Prepare example feedback items for analysis.
        
        Args:
            feedback_data: Raw feedback data
            limit: Maximum number of examples
            
        Returns:
            List of prepared examples
        """
        try:
            if not feedback_data:
                return []
            
            # Sort by most recent (if timestamp available)
            try:
                sorted_data = sorted(
                    feedback_data, 
                    key=lambda x: x.get("created_at", x.get("timestamp", 0)), 
                    reverse=True
                )
            except (TypeError, KeyError):
                # If sorting fails, just use as-is
                sorted_data = feedback_data
            
            # Get a mix of positive and negative examples
            positive = [item for item in sorted_data if item.get("score", 0) >= 0.5][:limit//2]
            negative = [item for item in sorted_data if item.get("score", 0) < 0.5][:limit//2]
            
            # Combine and format
            examples = []
            for item in positive + negative:
                try:
                    example = {
                        "query": self._extract_query(item),
                        "response": item.get("response", {}).get("content", item.get("response_content", "")),
                        "score": item.get("score")
                    }
                    
                    # Only include if we have meaningful data
                    if example["query"] or example["response"]:
                        examples.append(example)
                        
                except Exception as e:
                    logger.warning(f"Failed to process feedback example: {str(e)}")
                    continue
            
            return examples[:limit]  # Ensure we don't exceed limit
            
        except Exception as e:
            logger.error(f"Failed to prepare examples: {str(e)}")
            return []
    
    def _extract_query(self, feedback_item: Dict[str, Any]) -> str:
        """Extract the user query from feedback.
        
        Args:
            feedback_item: Individual feedback item
            
        Returns:
            Extracted query string
        """
        try:
            # Try multiple ways to extract the query
            if "formatted_prompt" in feedback_item:
                formatted_text = feedback_item["formatted_prompt"]
            elif "prompt_instance" in feedback_item:
                formatted_text = feedback_item["prompt_instance"].get("formatted_text", "")
            else:
                formatted_text = feedback_item.get("formatted_text", "")
            
            if not formatted_text:
                return ""
            
            # Simple heuristic to extract query from formatted prompt
            for prefix in ["Summarize the following text:", "Translate:", "Answer:", "Classify:", "Analyze:"]:
                if prefix in formatted_text:
                    return formatted_text.split(prefix, 1)[1].strip()
            
            return formatted_text
            
        except Exception as e:
            logger.warning(f"Failed to extract query: {str(e)}")
            return ""
    
    def _create_optimization_prompt(self, 
                                   current_prompt: str, 
                                   examples: List[Dict[str, Any]]) -> str:
        """Create a prompt asking AI to optimize the prompt template.
        
        Args:
            current_prompt: Current prompt text
            examples: Feedback examples
            
        Returns:
            Optimization prompt text
        """
        try:
            validate_not_empty(current_prompt, "Current prompt")
            
            ai_prompt = f"""You are an expert prompt engineer. Your task is to improve this prompt template:

CURRENT PROMPT TEMPLATE:
"{current_prompt}"

Based on the following examples, suggest an improved version of this prompt template.
The examples show queries sent to an AI system, the responses, and whether the user was satisfied.

EXAMPLES:
"""
            
            for i, example in enumerate(examples, 1):
                feedback_type = "👍 POSITIVE" if example.get("score", 0) >= 0.5 else "👎 NEGATIVE"
                ai_prompt += f"""
Example {i} ({feedback_type}):
Query: {example['query'][:200]}{'...' if len(example['query']) > 200 else ''}
Response: {example['response'][:200]}{'...' if len(example['response']) > 200 else ''}
"""
                if example.get("score") is not None:
                    ai_prompt += f"Score: {example['score']}/1.0\n"
            
            ai_prompt += """
Based on these examples, what patterns do you see in the responses users liked vs. disliked?
What issues need to be fixed in the prompt template?

Please provide an improved prompt template that addresses these issues. The improved template should:
1. Fix any clarity or specificity problems
2. Include better instructions or constraints
3. Maintain the same basic functionality and variables
4. Keep any placeholder variables in the format {variable_name}

IMPROVED PROMPT TEMPLATE:
"""
            
            return ai_prompt
            
        except Exception as e:
            logger.error(f"Failed to create optimization prompt: {str(e)}")
            raise OptimizationError(f"Failed to create optimization prompt: {str(e)}") from e
    
    def _extract_prompt(self, ai_response: str) -> str:
        """Extract the prompt from the AI's full response.
        
        Args:
            ai_response: Full AI response
            
        Returns:
            Extracted prompt text
        """
        try:
            if not ai_response:
                return ""
            
            # Look for the improved prompt section
            if "IMPROVED PROMPT TEMPLATE:" in ai_response:
                # Extract just the prompt part
                prompt_part = ai_response.split("IMPROVED PROMPT TEMPLATE:", 1)[1].strip()
                # Remove any analysis that might follow
                if "\n\n" in prompt_part:
                    prompt_part = prompt_part.split("\n\n", 1)[0]
                return prompt_part.strip()
            
            # If no clear section, just return the whole thing (cleaned up)
            return ai_response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract prompt from AI response: {str(e)}")
            return ai_response.strip() if ai_response else ""
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current strategy settings."""
        return {
            'name': self.name,
            'min_feedback_samples': self.min_feedback_samples,
            'min_positive_rate': self.min_positive_rate,
            'description': self.description
        }
    
    def update_settings(self, **new_settings):
        """Update strategy settings at runtime.
        
        Args:
            **new_settings: Settings to update
            
        Raises:
            OptimizationError: If settings update fails
        """
        try:
            updated = []
            
            if 'min_feedback_samples' in new_settings:
                samples = new_settings['min_feedback_samples']
                if not isinstance(samples, int) or samples < 1:
                    raise ValueError("min_feedback_samples must be a positive integer")
                self.min_feedback_samples = samples
                updated.append('min_feedback_samples')
                
            if 'min_positive_rate' in new_settings:
                rate = new_settings['min_positive_rate']
                if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                    raise ValueError("min_positive_rate must be a number between 0 and 1")
                self.min_positive_rate = rate
                updated.append('min_positive_rate')
            
            if updated:
                logger.info(f"SimpleAI strategy settings updated: {updated}")
                
        except ValueError as e:
            raise OptimizationError(f"Invalid setting value: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to update SimpleAI settings: {str(e)}")
            raise OptimizationError(f"Settings update failed: {str(e)}") from e