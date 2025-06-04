# prompt_optimizer/optimization_manager.py

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import os

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.core.prompt_manager import PromptManager
from prompt_optimizer.strategies.reward_model import RewardModel
from prompt_optimizer.strategies.shadow_testing import ShadowTester
from prompt_optimizer.strategies.bandit import ThompsonBandit
from prompt_optimizer.strategies.components.promotion import PromotionDecider
from prompt_optimizer.services.llm_service import LLMService

class OptimizationManager:
    """Main class that orchestrates the entire prompt optimization process."""
    
    def __init__(self,
                 prompt_manager: PromptManager,
                 llm_service: LLMService,
                 optimization_threshold: float = 0.05,
                 confidence_level: float = 0.95,
                 min_feedback_samples: int = 30,
                 model_dir: str = "./models",
                 log_dir: str = "./logs"):
        """Initialize the optimization manager.
        
        Args:
            prompt_manager: PromptManager instance for prompt operations
            llm_service: Service for generating LLM responses
            optimization_threshold: Minimum improvement required for promotion
            confidence_level: Statistical confidence required
            min_feedback_samples: Minimum samples needed for decisions
            model_dir: Directory to save models
            log_dir: Directory to save logs
        """
        self.prompt_manager = prompt_manager
        self.llm_service = llm_service
        self.model_dir = model_dir
        
        # Create component instances
        self.reward_model = RewardModel(model_dir=model_dir)
        self.promotion_decider = PromotionDecider(
            threshold=optimization_threshold,
            confidence=confidence_level,
            min_samples=min_feedback_samples
        )
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("prompt_optimizer")
        if not self.logger.handlers:
            handler = logging.FileHandler(
                os.path.join(log_dir, f"optimization_{datetime.now().strftime('%Y%m%d')}.log")
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info("Initialized OptimizationManager")
    
    def prepare_feedback_data(self, 
                              feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare feedback data for reward model training.
        
        Args:
            feedback_data: Raw feedback data from the database
            
        Returns:
            Processed feedback data ready for reward model training
        """
        processed_data = []
        
        for item in feedback_data:
            # Extract required fields
            prompt_instance = item.get("prompt_instance", {})
            response = item.get("response", {})
            
            prompt_text = prompt_instance.get("formatted_text", "")
            response_text = response.get("content", "")
            is_positive = item.get("score", 0) >= 0.5
            
            processed_data.append({
                "prompt": prompt_text,
                "response": response_text,
                "is_positive": is_positive
            })
        
        self.logger.info(f"Prepared {len(processed_data)} feedback items for training")
        return processed_data
    
    def train_reward_model(self, 
                           feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the reward model on historical feedback.
        
        Args:
            feedback_data: Prepared feedback data
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training reward model on {len(feedback_data)} examples")
        
        processed_data = self.prepare_feedback_data(feedback_data)
        metrics = self.reward_model.train(processed_data)
        
        self.logger.info(f"Reward model training completed: val_auc={metrics.get('val_auc', 0):.3f}")
        return metrics
    
    def generate_candidates(self, 
                           production_prompt: Prompt, 
                           num_candidates: int = 5) -> List[Prompt]:
        """Generate candidate prompts from the production prompt.
        
        Args:
            production_prompt: Current production prompt
            num_candidates: Number of candidates to generate
            
        Returns:
            List of candidate prompts
        """
        self.logger.info(f"Generating {num_candidates} candidate prompts")
        
        # Start with the current production prompt
        candidates = [production_prompt]
        
        # Simple candidate generation strategies:
        # 1. Add specificity
        specificity_prompt = Prompt(
            text=f"Be specific and detailed. {production_prompt.text}",
            description=f"Added specificity to {production_prompt.id}",
            version=production_prompt.version + 1,
            parent_id=production_prompt.id
        )
        candidates.append(specificity_prompt)
        
        # 2. Add structure
        structure_prompt = Prompt(
            text=f"Provide a well-structured response. {production_prompt.text}",
            description=f"Added structure to {production_prompt.id}",
            version=production_prompt.version + 1,
            parent_id=production_prompt.id
        )
        candidates.append(structure_prompt)
        
        # 3. Add conciseness
        concise_prompt = Prompt(
            text=f"Be concise and to the point. {production_prompt.text}",
            description=f"Added conciseness to {production_prompt.id}",
            version=production_prompt.version + 1,
            parent_id=production_prompt.id
        )
        candidates.append(concise_prompt)
        
        # 4. Add reasoning
        reasoning_prompt = Prompt(
            text=f"{production_prompt.text} Explain your reasoning step-by-step.",
            description=f"Added reasoning to {production_prompt.id}",
            version=production_prompt.version + 1,
            parent_id=production_prompt.id
        )
        candidates.append(reasoning_prompt)
        
        # More sophisticated generation would be implemented here
        # e.g., using an LLM to generate variations
        
        self.logger.info(f"Generated {len(candidates)} candidate prompts")
        return candidates
    
    def evaluate_candidates(self, 
                           candidates: List[Prompt], 
                           user_queries: List[str],
                           max_queries_per_candidate: int = 100) -> Dict[str, Any]:
        """Evaluate candidate prompts using shadow testing and bandit algorithm.
        
        Args:
            candidates: List of candidate prompts
            user_queries: Historical user queries
            max_queries_per_candidate: Maximum queries to evaluate per candidate
            
        Returns:
            Evaluation results
        """
        if not user_queries:
            raise ValueError("No user queries provided for evaluation")
        
        if not self.reward_model.is_trained:
            raise RuntimeError("Reward model must be trained before evaluation")
        
        self.logger.info(f"Evaluating {len(candidates)} candidates on {len(user_queries)} queries")
        
        # Create shadow tester
        shadow_tester = ShadowTester(self.llm_service, self.reward_model)
        
        # Create bandit
        bandit = ThompsonBandit(candidates)
        
        # Select a reasonable subset of queries to evaluate
        if len(user_queries) > max_queries_per_candidate:
            np.random.shuffle(user_queries)
            evaluation_queries = user_queries[:max_queries_per_candidate]
        else:
            evaluation_queries = user_queries
        
        # Run shadow evaluations through the bandit
        for query in evaluation_queries:
            # Select arm using Thompson sampling
            arm = bandit.select_arm()
            
            # Generate shadow response
            formatted_prompt = shadow_tester._format_prompt(arm.prompt.text, query)
            response = self.llm_service.generate(formatted_prompt)
            
            # Score with reward model
            reward = self.reward_model.predict(formatted_prompt, response)
            
            # Update bandit
            bandit.update(arm, reward)
        
        # Get best candidate
        best_arm = bandit.best_arm()
        
        # Get statistics for all candidates
        arm_stats = bandit.get_arm_stats()
        
        self.logger.info(f"Evaluation complete. Best candidate: {best_arm.prompt.id} with estimate {best_arm.mean_estimate:.3f}")
        
        return {
            "best_candidate": {
                "prompt_id": best_arm.prompt.id,
                "prompt_text": best_arm.prompt.text,
                "mean_estimate": best_arm.mean_estimate,
                "total_pulls": best_arm.total_pulls
            },
            "all_candidates": arm_stats,
            "queries_evaluated": len(evaluation_queries)
        }
    
    def make_promotion_decision(self, 
                              production_stats: Dict[str, Any],
                              candidate_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to promote a candidate to production.
        
        Args:
            production_stats: Statistics for production prompt
            candidate_stats: Statistics for best candidate prompt
            
        Returns:
            Decision results
        """
        self.logger.info("Making promotion decision")
        
        decision = self.promotion_decider.evaluate_promotion(
            production_stats, candidate_stats
        )
        
        if decision["decision"] == "promote":
            self.logger.info(f"Decision: PROMOTE candidate. Reason: {decision['reason']}")
        else:
            self.logger.info(f"Decision: NO PROMOTION. Reason: {decision['reason']}")
        
        return decision
    
    def run_optimization_cycle(self, 
                             production_prompt_id: str,
                             feedback_data: List[Dict[str, Any]],
                             user_queries: List[str]) -> Dict[str, Any]:
        """Run a complete optimization cycle.
        
        Args:
            production_prompt_id: ID of current production prompt
            feedback_data: Historical feedback data
            user_queries: Historical user queries
            
        Returns:
            Results of the optimization cycle
        """
        self.logger.info(f"Starting optimization cycle for prompt {production_prompt_id}")
        
        # Get production prompt
        production_prompt = self.prompt_manager.get_prompt(production_prompt_id)
        if not production_prompt:
            raise ValueError(f"Production prompt {production_prompt_id} not found")
        
        # 1. Train reward model
        training_metrics = self.train_reward_model(feedback_data)
        
        # 2. Generate candidates
        candidates = self.generate_candidates(production_prompt)
        
        # 3. Evaluate candidates
        evaluation_results = self.evaluate_candidates(candidates, user_queries)
        
        # 4. Extract production statistics from feedback data
        production_stats = self._calculate_production_stats(feedback_data, production_prompt_id)
        
        # 5. Make promotion decision
        best_candidate_stats = {
            "successes": int(evaluation_results["best_candidate"]["mean_estimate"] * 
                            evaluation_results["best_candidate"]["total_pulls"]),
            "total": evaluation_results["best_candidate"]["total_pulls"]
        }
        
        decision = self.make_promotion_decision(production_stats, best_candidate_stats)
        
        # 6. Promote if decision is positive
        new_production_id = None
        if decision["decision"] == "promote":
            best_candidate_id = evaluation_results["best_candidate"]["prompt_id"]
            best_candidate = next((c for c in candidates if c.id == best_candidate_id), None)
            
            if best_candidate:
                # Save the promoted prompt
                saved_prompt = self.prompt_manager.update_prompt(
                    prompt_id=production_prompt_id,
                    text=best_candidate.text,
                    description=f"Optimized from {production_prompt_id}"
                )
                new_production_id = saved_prompt.id
                
                self.logger.info(f"Promoted {best_candidate_id} to production as {new_production_id}")
        
        return {
            "production_prompt_id": production_prompt_id,
            "training_metrics": training_metrics,
            "evaluation_results": evaluation_results,
            "decision": decision,
            "new_production_id": new_production_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_production_stats(self, 
                                  feedback_data: List[Dict[str, Any]],
                                  prompt_id: str) -> Dict[str, Any]:
        """Calculate statistics for the production prompt from feedback data.
        
        Args:
            feedback_data: Historical feedback data
            prompt_id: ID of the prompt to calculate statistics for
            
        Returns:
            Statistics for the production prompt
        """
        # Filter feedback for the specified prompt
        prompt_feedback = [
            item for item in feedback_data 
            if item.get("prompt_instance", {}).get("prompt_id") == prompt_id
        ]
        
        total = len(prompt_feedback)
        successes = sum(1 for item in prompt_feedback if item.get("score", 0) >= 0.5)
        
        return {
            "prompt_id": prompt_id,
            "total": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0
        }