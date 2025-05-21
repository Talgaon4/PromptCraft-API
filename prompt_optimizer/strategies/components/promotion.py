# prompt_optimizer/strategies/promotion.py

import numpy as np
from typing import Dict, Any, Optional, Tuple
import scipy.stats as stats

class PromotionDecider:
    """Decides when to promote a candidate prompt to production."""
    
    def __init__(self, 
                 threshold: float = 0.05,  # 5% improvement required
                 confidence: float = 0.95,  # 95% confidence
                 min_samples: int = 30):     # Minimum samples needed
        """Initialize the promotion decider.
        
        Args:
            threshold: Minimum improvement required for promotion
            confidence: Statistical confidence required
            min_samples: Minimum number of samples required
        """
        self.threshold = threshold
        self.confidence = confidence
        self.min_samples = min_samples
    
    def is_ready_for_comparison(self, 
                              production_samples: int, 
                              candidate_samples: int) -> bool:
        """Check if we have enough data to make a promotion decision.
        
        Args:
            production_samples: Number of feedback samples for production prompt
            candidate_samples: Number of (estimated) samples for candidate
            
        Returns:
            True if ready for comparison, False otherwise
        """
        return (production_samples >= self.min_samples and 
                candidate_samples >= self.min_samples)
    
    def evaluate_promotion(self, 
                          production_stats: Dict[str, Any],
                          candidate_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether to promote a candidate prompt.
        
        Args:
            production_stats: Statistics for production prompt
            candidate_stats: Statistics for candidate prompt
            
        Returns:
            Evaluation result with decision and metrics
        """
        # Extract statistics
        prod_success = production_stats.get("successes", 0)
        prod_total = production_stats.get("total", 0)
        
        cand_success = candidate_stats.get("successes", 0)
        cand_total = candidate_stats.get("total", 0)
        
        # Check if we have enough data
        if not self.is_ready_for_comparison(prod_total, cand_total):
            return {
                "decision": "insufficient_data",
                "reason": f"Need at least {self.min_samples} samples for each prompt",
                "production_samples": prod_total,
                "candidate_samples": cand_total
            }
        
        # Calculate success rates
        prod_rate = prod_success / prod_total if prod_total > 0 else 0
        cand_rate = cand_success / cand_total if cand_total > 0 else 0
        
        # Calculate improvement
        improvement = cand_rate - prod_rate
        
        # Perform statistical test (Binomial proportion test)
        z_score, p_value = self._proportion_test(
            prod_success, prod_total, cand_success, cand_total
        )
        
        # Decision logic
        is_significant = p_value < (1 - self.confidence)
        meets_threshold = improvement >= self.threshold
        
        if is_significant and meets_threshold:
            decision = "promote"
            reason = f"Significant improvement of {improvement:.1%}"
        elif not is_significant:
            decision = "no_promotion"
            reason = f"Improvement not statistically significant (p={p_value:.3f})"
        elif not meets_threshold:
            decision = "no_promotion"
            reason = f"Improvement ({improvement:.1%}) below threshold ({self.threshold:.1%})"
        else:
            decision = "no_promotion"
            reason = "Unknown reason"
        
        return {
            "decision": decision,
            "reason": reason,
            "production_rate": prod_rate,
            "candidate_rate": cand_rate,
            "improvement": improvement,
            "p_value": p_value,
            "is_significant": is_significant,
            "meets_threshold": meets_threshold
        }
    
    def _proportion_test(self, 
                       count1: int, n1: int, 
                       count2: int, n2: int) -> Tuple[float, float]:
        """Perform a statistical test comparing two proportions.
        
        Args:
            count1: Number of successes in first group
            n1: Total number in first group
            count2: Number of successes in second group
            n2: Total number in second group
            
        Returns:
            Tuple of (z_score, p_value)
        """
        # Calculate proportions
        p1 = count1 / n1 if n1 > 0 else 0
        p2 = count2 / n2 if n2 > 0 else 0
        
        # Calculate pooled proportion
        p_pooled = (count1 + count2) / (n1 + n2)
        
        # Calculate standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Handle division by zero
        if se == 0:
            return 0, 1.0
        
        # Calculate z-score
        z = (p2 - p1) / se
        
        # Calculate p-value (one-sided test since we only care if candidate is better)
        p = 1 - stats.norm.cdf(z)
        
        return z, p