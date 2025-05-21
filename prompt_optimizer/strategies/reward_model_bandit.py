# prompt_optimizer/strategies/reward_model_bandit.py

from typing import List, Dict, Any, Optional
import os
import logging
from datetime import datetime

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.strategies.base_strategy import OptimizationStrategy
from prompt_optimizer.strategies.components.reward_model import RewardModel
from prompt_optimizer.strategies.components.shadow_testing import ShadowTester
from prompt_optimizer.strategies.components.bandit import ThompsonBandit
from prompt_optimizer.strategies.components.promotion import PromotionDecider
from prompt_optimizer.services.llm_service import LLMService

class RewardModelBanditStrategy(OptimizationStrategy):
    """Advanced optimization using reward model and bandit algorithm."""
    
    def __init__(self, 
                 llm_service: LLMService,
                 model_dir: str = "./models",
                 optimization_threshold: float = 0.05,
                 min_feedback_samples: int = 30,
                 confidence_level: float = 0.95,
                 validation_size: float = 0.2,
                 max_candidates: int = 5,
                 max_queries_per_candidate: int = 100):
        """Initialize the strategy.
        
        Args:
            llm_service: Service for generating LLM responses
            model_dir: Directory for storing models
            optimization_threshold: Minimum improvement required
            min_feedback_samples: Minimum feedback samples needed
            confidence_level: Statistical confidence required
            validation_size: Portion of data to use for validation
            max_candidates: Maximum number of candidates to evaluate
            max_queries_per_candidate: Maximum queries per candidate
        """
        self.llm_service = llm_service
        self.model_dir = model_dir
        self.min_feedback_samples = min_feedback_samples
        self.max_candidates = max_candidates
        self.max_queries_per_candidate = max_queries_per_candidate
        
        # Create component instances
        self.reward_model = RewardModel(
            model_dir=model_dir,
            validation_size=validation_size
        )
        
        self.promotion_decider = PromotionDecider(
            threshold=optimization_threshold,
            confidence=confidence_level,
            min_samples=min_feedback_samples
        )
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("reward_model_bandit")
    
    @property
    def name(self) -> str:
        return "reward_model_bandit"
    
    @property
    def description(self) -> str:
        return "Advanced ML-based strategy using reward modeling and bandit optimization"
    
    def is_ready_for_optimization(self, 
                                feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if there's enough data to run optimization."""
        total_feedback = len(feedback_data)
        
        if total_feedback < self.min_feedback_samples:
            return {
                "ready": False,
                "reason": f"Insufficient feedback: {total_feedback}/{self.min_feedback_samples}",
                "total_feedback": total_feedback
            }
        
        # Check if there's enough variety in the feedback
        positive = sum(1 for item in feedback_data if item.get("is_positive", False))
        negative = total_feedback - positive
        
        # Need some of both for a good model
        if positive < 3 or negative < 3:
            return {
                "ready": False,
                "reason": f"Need more diverse feedback: {positive} positive, {negative} negative",
                "total_feedback": total_feedback,
                "positive_count": positive,
                "negative_count": negative
            }
        
        return {
            "ready": True,
            "reason": "Sufficient feedback available",
            "total_feedback": total_feedback,
            "positive_count": positive,
            "negative_count": negative
        }
    
    def optimize(self, 
               production_prompt: Prompt,
               feedback_data: List[Dict[str, Any]],
               user_queries: List[str],
               **kwargs) -> Dict[str, Any]:
        """Run the optimization process."""
        # Check if ready
        readiness = self.is_ready_for_optimization(feedback_data)
        if not readiness["ready"]:
            return {
                "status": "not_optimized",
                "reason": readiness["reason"],
                "new_prompt": None
            }
        
        # 1. Train reward model
        processed_data = self._prepare_feedback_data(feedback_data)
        training_metrics = self.reward_model.train(processed_data)
        
        if training_metrics.get("val_auc", 0) < 0.6:
            return {
                "status": "not_optimized",
                "reason": f"Reward model performance too low: AUC = {training_metrics.get('val_auc', 0):.2f}",
                "new_prompt": None,
                "training_metrics": training_metrics
            }
        
        # 2. Generate candidates
        candidates = self._generate_candidates(production_prompt)
        
        # 3. Shadow testing with bandit
        shadow_tester = ShadowTester(self.llm_service, self.reward_model)
        bandit = ThompsonBandit(candidates)
        
        # Limit number of queries to evaluate
        if len(user_queries) > self.max_queries_per_candidate:
            import random
            random.shuffle(user_queries)
            evaluation_queries = user_queries[:self.max_queries_per_candidate]
        else:
            evaluation_queries = user_queries
        
        # Run evaluations through bandit
        for query in evaluation_queries:
            # Select arm using Thompson sampling
            arm = bandit.select_arm()
            
            # Generate shadow response
            formatted_prompt = self._format_prompt(arm.prompt.text, query)
            response = self.llm_service.generate(formatted_prompt)
            
            # Score with reward model
            reward = self.reward_model.predict(formatted_prompt, response)
            
            # Update bandit
            bandit.update(arm, reward)
        
        # 4. Get best candidate
        best_arm = bandit.best_arm()
        
        # 5. Make promotion decision
        production_stats = self._calculate_production_stats(feedback_data)
        
        best_candidate_stats = {
            "successes": int(best_arm.mean_estimate * best_arm.total_pulls),
            "total": best_arm.total_pulls
        }
        
        decision = self.promotion_decider.evaluate_promotion(
            production_stats, best_candidate_stats
        )
        
        if decision["decision"] == "promote":
            return {
                "status": "optimized",
                "reason": decision["reason"],
                "new_prompt": best_arm.prompt.text,
                "original_prompt": production_prompt.text,
                "improvement": f"{decision.get('improvement', 0):.1%}",
                "training_metrics": training_metrics,
                "p_value": decision.get("p_value")
            }
        else:
            return {
                "status": "not_optimized",
                "reason": decision["reason"],
                "new_prompt": None,
                "training_metrics": training_metrics
            }
    
    def _prepare_feedback_data(self, 
                              feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare feedback data for reward model training."""
        processed_data = []
        
        for item in feedback_data:
            # Extract required fields
            prompt_instance = item.get("prompt_instance", {})
            response = item.get("response", {})
            
            prompt_text = prompt_instance.get("formatted_text", "")
            response_text = response.get("content", "")
            is_positive = item.get("is_positive", False)
            
            processed_data.append({
                "prompt": prompt_text,
                "response": response_text,
                "is_positive": is_positive
            })
        
        return processed_data
    
    def _generate_candidates(self, prompt: Prompt) -> List[Prompt]:
        """Generate candidate prompts for optimization."""
        # Start with the current production prompt
        candidates = [prompt]
        
        # Simple candidate generation strategies:
        # 1. Add specificity
        specificity_prompt = Prompt(
            text=f"Be specific and detailed. {prompt.text}",
            description=f"Added specificity to {prompt.id}",
            version=prompt.version + 1,
            parent_id=prompt.id
        )
        candidates.append(specificity_prompt)
        
        # 2. Add structure
        structure_prompt = Prompt(
            text=f"Provide a well-structured response. {prompt.text}",
            description=f"Added structure to {prompt.id}",
            version=prompt.version + 1,
            parent_id=prompt.id
        )
        candidates.append(structure_prompt)
        
        # 3. Add conciseness
        concise_prompt = Prompt(
            text=f"Be concise and to the point. {prompt.text}",
            description=f"Added conciseness to {prompt.id}",
            version=prompt.version + 1,
            parent_id=prompt.id
        )
        candidates.append(concise_prompt)
        
        # 4. Add reasoning
        reasoning_prompt = Prompt(
            text=f"{prompt.text} Explain your reasoning step-by-step.",
            description=f"Added reasoning to {prompt.id}",
            version=prompt.version + 1,
            parent_id=prompt.id
        )
        candidates.append(reasoning_prompt)
        
        return candidates[:self.max_candidates]
    
    def _format_prompt(self, prompt_template: str, query: str) -> str:
        """Format a prompt template with a user query."""
        # This is a simplified implementation
        # You'll need to adapt this to your prompt format
        return prompt_template.replace("{input_text}", query)
    
    def _calculate_production_stats(self, 
                                  feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for the production prompt."""
        total = len(feedback_data)
        successes = sum(1 for item in feedback_data if item.get("is_positive", False))
        
        return {
            "total": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0
        }