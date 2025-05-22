# prompt_optimizer/strategies/reward_model_bandit.py

"""Advanced optimization using reward model and bandit algorithm with configuration support."""

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
from prompt_optimizer.config import config

class RewardModelBanditStrategy(OptimizationStrategy):
    """Advanced optimization using reward model and bandit algorithm with configurable settings."""
    
    def __init__(self, 
                 llm_service: LLMService,
                 config_instance=None,
                 **overrides):
        """Initialize the strategy.
        
        Args:
            llm_service: Service for generating LLM responses
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides:
                - model_dir: Directory for storing models
                - optimization_threshold: Minimum improvement required
                - min_feedback_samples: Minimum feedback samples needed
                - confidence_level: Statistical confidence required
                - validation_size: Portion of data for validation
                - max_candidates: Maximum number of candidates to evaluate
                - max_queries_per_candidate: Maximum queries per candidate
        """
        self.llm_service = llm_service
        
        # Handle configuration
        self.config = config_instance or config
        
        # Apply overrides or use config values
        self.model_dir = overrides.get('model_dir', 
                                      os.path.join(self.config.DEFAULT_STORAGE_DIR, 'models'))
        self.min_feedback_samples = overrides.get('min_feedback_samples', 
                                                 self.config.REWARD_MODEL_MIN_SAMPLES)
        self.max_candidates = overrides.get('max_candidates', 
                                           self.config.REWARD_MODEL_MAX_CANDIDATES)
        self.max_queries_per_candidate = overrides.get('max_queries_per_candidate', 
                                                      getattr(self.config, 'REWARD_MODEL_MAX_QUERIES', 100))
        self.optimization_threshold = overrides.get('optimization_threshold', 
                                                   getattr(self.config, 'IMPROVEMENT_THRESHOLD', 0.05))
        self.confidence_level = overrides.get('confidence_level', 
                                             self.config.CONFIDENCE_LEVEL)
        self.validation_size = overrides.get('validation_size', 
                                            getattr(self.config, 'REWARD_MODEL_VALIDATION_SIZE', 0.2))
        
        # Create component instances
        self.reward_model = RewardModel(
            model_dir=self.model_dir,
            validation_size=self.validation_size
        )
        
        self.promotion_decider = PromotionDecider(
            threshold=self.optimization_threshold,
            confidence=self.confidence_level,
            min_samples=self.min_feedback_samples
        )
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
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
                "total_feedback": total_feedback,
                "required_samples": self.min_feedback_samples
            }
        
        # Check if there's enough variety in the feedback
        positive = sum(1 for item in feedback_data if item.get("is_positive", False))
        negative = total_feedback - positive
        
        # Need some of both for a good model
        min_positive = max(3, int(self.min_feedback_samples * 0.2))  # At least 20% positive or 3
        min_negative = max(3, int(self.min_feedback_samples * 0.2))  # At least 20% negative or 3
        
        if positive < min_positive or negative < min_negative:
            return {
                "ready": False,
                "reason": f"Need more diverse feedback: {positive} positive (need {min_positive}), {negative} negative (need {min_negative})",
                "total_feedback": total_feedback,
                "positive_count": positive,
                "negative_count": negative,
                "required_positive": min_positive,
                "required_negative": min_negative
            }
        
        return {
            "ready": True,
            "reason": "Sufficient diverse feedback available",
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
                "new_prompt": None,
                "readiness_info": readiness
            }
        
        try:
            # 1. Train reward model
            processed_data = self._prepare_feedback_data(feedback_data)
            training_metrics = self.reward_model.train(processed_data)
            
            # Check if model training was successful
            min_auc = getattr(self.config, 'REWARD_MODEL_MIN_AUC', 0.6)
            if training_metrics.get("val_auc", 0) < min_auc:
                return {
                    "status": "not_optimized",
                    "reason": f"Reward model performance too low: AUC = {training_metrics.get('val_auc', 0):.2f} (need {min_auc})",
                    "new_prompt": None,
                    "training_metrics": training_metrics
                }
            
            # 2. Generate candidates
            candidates = self._generate_candidates(production_prompt)
            
            # 3. Shadow testing with bandit
            shadow_tester = ShadowTester(self.llm_service, self.reward_model)
            bandit = ThompsonBandit(candidates)
            
            # Limit number of queries to evaluate
            evaluation_queries = self._select_evaluation_queries(user_queries)
            
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
                    "evaluation_info": {
                        "candidates_tested": len(candidates),
                        "queries_used": len(evaluation_queries),
                        "best_candidate_pulls": best_arm.total_pulls,
                        "best_candidate_score": best_arm.mean_estimate
                    },
                    "p_value": decision.get("p_value")
                }
            else:
                return {
                    "status": "not_optimized",
                    "reason": decision["reason"],
                    "new_prompt": None,
                    "training_metrics": training_metrics,
                    "evaluation_info": {
                        "candidates_tested": len(candidates),
                        "queries_used": len(evaluation_queries),
                        "best_candidate_score": best_arm.mean_estimate
                    }
                }
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return {
                "status": "error",
                "reason": f"Optimization failed: {str(e)}",
                "new_prompt": None
            }
    
    def _prepare_feedback_data(self, 
                              feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare feedback data for reward model training."""
        processed_data = []
        
        for item in feedback_data:
            # Extract required fields with fallbacks
            if "formatted_prompt" in item:
                prompt_text = item["formatted_prompt"]
            elif "prompt_instance" in item and "formatted_text" in item["prompt_instance"]:
                prompt_text = item["prompt_instance"]["formatted_text"]
            else:
                prompt_text = ""
            
            if "response_content" in item:
                response_text = item["response_content"]
            elif "response" in item and "content" in item["response"]:
                response_text = item["response"]["content"]
            else:
                response_text = ""
            
            is_positive = item.get("is_positive", False)
            
            if prompt_text and response_text:  # Only include if we have both
                processed_data.append({
                    "prompt": prompt_text,
                    "response": response_text,
                    "is_positive": is_positive
                })
        
        return processed_data
    
    def _generate_candidates(self, prompt: Prompt) -> List[Prompt]:
        """Generate candidate prompts for optimization."""
        candidates = [prompt]  # Include original
        
        # Get number of candidates from config
        num_additional = self.max_candidates - 1
        
        # Generate different types of candidate modifications
        modifications = [
            ("Be specific and detailed. ", "Added specificity"),
            ("Provide a well-structured response. ", "Added structure requirement"),
            ("Be concise and to the point. ", "Added conciseness requirement"),
            (" Explain your reasoning step-by-step.", "Added reasoning requirement"),
            ("Focus on accuracy. ", "Added accuracy emphasis"),
            ("Provide examples when helpful. ", "Added examples instruction")
        ]
        
        # Create candidates with different modifications
        for i, (modification, description) in enumerate(modifications[:num_additional]):
            if modification.endswith(" "):
                # Prefix modification
                new_text = modification + prompt.text
            else:
                # Suffix modification
                new_text = prompt.text + modification
            
            candidate = Prompt(
                text=new_text,
                description=f"{description} - {prompt.description}" if prompt.description else description,
                version=prompt.version + 1,
                parent_id=prompt.id
            )
            candidates.append(candidate)
        
        return candidates[:self.max_candidates]
    
    def _select_evaluation_queries(self, user_queries: List[str]) -> List[str]:
        """Select queries for evaluation based on config limits."""
        if len(user_queries) <= self.max_queries_per_candidate:
            return user_queries
        
        # Randomly sample queries
        import random
        sampled_queries = random.sample(user_queries, self.max_queries_per_candidate)
        return sampled_queries
    
    def _format_prompt(self, prompt_template: str, query: str) -> str:
        """Format a prompt template with a user query."""
        # Simple placeholder replacement
        if "{input_text}" in prompt_template:
            return prompt_template.replace("{input_text}", query)
        elif "{query}" in prompt_template:
            return prompt_template.replace("{query}", query)
        else:
            # If no clear placeholder, append query
            return f"{prompt_template}\n\nInput: {query}"
    
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
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current strategy settings."""
        return {
            'name': self.name,
            'model_dir': self.model_dir,
            'min_feedback_samples': self.min_feedback_samples,
            'max_candidates': self.max_candidates,
            'max_queries_per_candidate': self.max_queries_per_candidate,
            'optimization_threshold': self.optimization_threshold,
            'confidence_level': self.confidence_level,
            'validation_size': self.validation_size,
            'description': self.description
        }
    
    def update_settings(self, **new_settings):
        """Update strategy settings at runtime."""
        updated = []
        
        if 'min_feedback_samples' in new_settings:
            self.min_feedback_samples = new_settings['min_feedback_samples']
            # Update promotion decider
            self.promotion_decider.min_samples = self.min_feedback_samples
            updated.append('min_feedback_samples')
            
        if 'max_candidates' in new_settings:
            self.max_candidates = new_settings['max_candidates']
            updated.append('max_candidates')
            
        if 'max_queries_per_candidate' in new_settings:
            self.max_queries_per_candidate = new_settings['max_queries_per_candidate']
            updated.append('max_queries_per_candidate')
            
        if 'optimization_threshold' in new_settings:
            self.optimization_threshold = new_settings['optimization_threshold']
            self.promotion_decider.threshold = self.optimization_threshold
            updated.append('optimization_threshold')
            
        if 'confidence_level' in new_settings:
            self.confidence_level = new_settings['confidence_level']
            self.promotion_decider.confidence = self.confidence_level
            updated.append('confidence_level')
        
        if updated:
            self.logger.info(f"RewardModelBandit strategy settings updated: {updated}")