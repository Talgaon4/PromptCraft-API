# prompt_optimizer/strategies/simple_ai_strategy.py

from typing import List, Dict, Any, Optional
from prompt_optimizer.core.models import Prompt
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy
from prompt_optimizer.services.llm_service import LLMService

class SimpleAIStrategy(OptimizationStrategy):
    """Simple prompt optimization using a language model."""
    
    def __init__(self, 
                 llm_service: LLMService,
                 min_feedback_samples: int = 5,
                 min_positive_rate: float = 0.8):
        """Initialize the simple AI strategy.
        
        Args:
            llm_service: Service for generating LLM responses
            min_feedback_samples: Minimum feedback samples required
            min_positive_rate: Minimum positive feedback rate to trigger optimization
        """
        self.llm_service = llm_service
        self.min_feedback_samples = min_feedback_samples
        self.min_positive_rate = min_positive_rate
    
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
                "reason": f"Insufficient feedback: {total_feedback}/{self.min_feedback_samples}"
            }
        
        # Calculate positive rate
        positive_count = sum(1 for item in feedback_data if item.get("is_positive", False))
        positive_rate = positive_count / total_feedback if total_feedback > 0 else 0
        
        # If positive rate is already high, maybe no need to optimize
        if positive_rate >= 0.9:
            return {
                "ready": False, 
                "reason": f"Prompt is already performing well: {positive_rate:.1%} positive"
            }
        
        # If positive rate is too low, optimization is needed
        needs_optimization = positive_rate < self.min_positive_rate
        
        return {
            "ready": needs_optimization,
            "reason": f"Prompt needs improvement: {positive_rate:.1%} positive" if needs_optimization else f"Prompt is performing adequately: {positive_rate:.1%} positive",
            "positive_rate": positive_rate,
            "total_feedback": total_feedback
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
                "new_prompt": None
            }
        
        # Prepare examples for the AI
        examples = self._prepare_examples(feedback_data, limit=5)
        
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
            "examples_analyzed": len(examples)
        }
    
    def _prepare_examples(self, 
                         feedback_data: List[Dict[str, Any]], 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """Prepare example feedback items for analysis."""
        # Sort by most recent
        sorted_data = sorted(
            feedback_data, 
            key=lambda x: x.get("timestamp", 0), 
            reverse=True
        )
        
        # Get a mix of positive and negative examples
        positive = [item for item in sorted_data if item.get("is_positive", False)][:limit//2]
        negative = [item for item in sorted_data if not item.get("is_positive", False)][:limit//2]
        
        # Combine and format
        examples = []
        for item in positive + negative:
            examples.append({
                "query": self._extract_query(item),
                "response": item.get("response", {}).get("content", ""),
                "is_positive": item.get("is_positive", False),
                "comments": item.get("comments", "")
            })
        
        return examples
    
    def _extract_query(self, feedback_item: Dict[str, Any]) -> str:
        """Extract the user query from feedback."""
        # In a real implementation, extract the actual query
        # This is a simplified placeholder
        prompt_instance = feedback_item.get("prompt_instance", {})
        formatted_text = prompt_instance.get("formatted_text", "")
        
        # Simple heuristic to extract query
        for prefix in ["Summarize the following text:", "Translate:", "Answer:"]:
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
Response: {example['response']}
"""
            if example.get("comments"):
                ai_prompt += f"User Comments: {example['comments']}\n"
        
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
        
        # If no clear section, just return the whole thing
        return ai_response