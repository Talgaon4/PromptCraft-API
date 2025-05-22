# prompt_optimizer/strategies/simple_ai_strategy.py

"""Simple prompt optimization using a language model with configuration support."""

from typing import List, Dict, Any, Optional
from prompt_optimizer.core.models import Prompt
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.config import config

class SimpleAIStrategy(OptimizationStrategy):
    """Simple prompt optimization using a language model with configurable settings."""
    
    def __init__(self, 
                 llm_service: LLMService,
                 config_instance=None,
                 **overrides):
        """Initialize the simple AI strategy.
        
        Args:
            llm_service: Service for generating LLM responses
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides:
                - min_feedback_samples: Minimum feedback samples required
                - min_positive_rate: Minimum positive feedback rate to trigger optimization
        """
        self.llm_service = llm_service
        
        # Handle configuration
        self.config = config_instance or config
        
        # Apply overrides or use config values
        self.min_feedback_samples = overrides.get('min_feedback_samples', 
                                                 self.config.SIMPLE_AI_MIN_SAMPLES)
        self.min_positive_rate = overrides.get('min_positive_rate', 
                                              self.config.SIMPLE_AI_MIN_POSITIVE_RATE)
    
    @property
    def name(self) -> str:
        return "simple_ai"
    
    @property
    def description(self) -> str:
        return "Optimize prompts by asking an AI to analyze feedback and suggest improvements"
    
    def is_ready_for_optimization(self, 
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if there's enough feedback to optimize."""
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
        positive_count = sum(1 for item in feedback_data if item.get("is_positive", False))
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
    
    def optimize(self, 
               production_prompt: Prompt,
               feedback_data: List[Dict[str, Any]],
               user_queries: List[str],
               **kwargs) -> Dict[str, Any]:
        """Optimize the prompt using AI."""
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
        
        # Create the optimization prompt
        optimization_prompt = self._create_optimization_prompt(
            current_prompt=production_prompt.text,
            examples=examples
        )
        
        # Get optimization from AI
        improved_prompt = self.llm_service.generate(optimization_prompt)
        
        # Clean up the response to extract just the prompt
        improved_prompt = self._extract_prompt(improved_prompt)
        
        return {
            "status": "optimized",
            "reason": "AI suggestion applied",
            "original_prompt": production_prompt.text,
            "new_prompt": improved_prompt,
            "examples_analyzed": len(examples),
            "readiness_info": readiness
        }
    
    def _prepare_examples(self, 
                         feedback_data: List[Dict[str, Any]], 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Prepare example feedback items for analysis."""
        # Sort by most recent (if timestamp available)
        try:
            sorted_data = sorted(
                feedback_data, 
                key=lambda x: x.get("created_at", x.get("timestamp", 0)), 
                reverse=True
            )
        except:
            # If sorting fails, just use as-is
            sorted_data = feedback_data
        
        # Get a mix of positive and negative examples
        positive = [item for item in sorted_data if item.get("is_positive", False)][:limit//2]
        negative = [item for item in sorted_data if not item.get("is_positive", False)][:limit//2]
        
        # Combine and format
        examples = []
        for item in positive + negative:
            examples.append({
                "query": self._extract_query(item),
                "response": item.get("response", {}).get("content", item.get("response_content", "")),
                "is_positive": item.get("is_positive", False),
                "comments": item.get("comments", ""),
                "score": item.get("score")
            })
        
        return examples
    
    def _extract_query(self, feedback_item: Dict[str, Any]) -> str:
        """Extract the user query from feedback."""
        # Try multiple ways to extract the query
        if "formatted_prompt" in feedback_item:
            formatted_text = feedback_item["formatted_prompt"]
        elif "prompt_instance" in feedback_item:
            formatted_text = feedback_item["prompt_instance"].get("formatted_text", "")
        else:
            formatted_text = feedback_item.get("formatted_text", "")
        
        # Simple heuristic to extract query from formatted prompt
        for prefix in ["Summarize the following text:", "Translate:", "Answer:", "Classify:", "Analyze:"]:
            if prefix in formatted_text:
                return formatted_text.split(prefix, 1)[1].strip()
        
        return formatted_text
    
    def _create_optimization_prompt(self, 
                                   current_prompt: str, 
                                   examples: List[Dict[str, Any]]) -> str:
        """Create a prompt asking AI to optimize the prompt template."""
        ai_prompt = f"""You are an expert prompt engineer. Your task is to improve this prompt template:

CURRENT PROMPT TEMPLATE:
"{current_prompt}"

Based on the following examples, suggest an improved version of this prompt template.
The examples show queries sent to an AI system, the responses, and whether the user was satisfied.

EXAMPLES:
"""
        
        for i, example in enumerate(examples, 1):
            feedback_type = "👍 POSITIVE" if example["is_positive"] else "👎 NEGATIVE"
            ai_prompt += f"""
Example {i} ({feedback_type}):
Query: {example['query']}
Response: {example['response'][:200]}{'...' if len(example['response']) > 200 else ''}
"""
            if example.get("comments"):
                ai_prompt += f"User Comments: {example['comments']}\n"
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
    
    def _extract_prompt(self, ai_response: str) -> str:
        """Extract the prompt from the AI's full response."""
        # Look for the improved prompt section
        if "IMPROVED PROMPT TEMPLATE:" in ai_response:
            # Extract just the prompt part
            prompt_part = ai_response.split("IMPROVED PROMPT TEMPLATE:", 1)[1].strip()
            # Remove any analysis that might follow
            if "\n\n" in prompt_part:
                prompt_part = prompt_part.split("\n\n", 1)[0]
            return prompt_part
        
        # If no clear section, just return the whole thing (cleaned up)
        return ai_response.strip()
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current strategy settings."""
        return {
            'name': self.name,
            'min_feedback_samples': self.min_feedback_samples,
            'min_positive_rate': self.min_positive_rate,
            'description': self.description
        }
    
    def update_settings(self, **new_settings):
        """Update strategy settings at runtime."""
        updated = []
        
        if 'min_feedback_samples' in new_settings:
            self.min_feedback_samples = new_settings['min_feedback_samples']
            updated.append('min_feedback_samples')
            
        if 'min_positive_rate' in new_settings:
            self.min_positive_rate = new_settings['min_positive_rate']
            updated.append('min_positive_rate')
        
        if updated:
            print(f"SimpleAI strategy settings updated: {updated}")