# prompt_optimizer/strategies/bandit.py

import numpy as np
from typing import List, Dict, Any, Optional
from prompt_optimizer.core.models import Prompt

class BanditArm:
    """Represents a candidate prompt in the bandit algorithm."""
    
    def __init__(self, prompt: Prompt):
        """Initialize a bandit arm for a candidate prompt.
        
        Args:
            prompt: The candidate prompt
        """
        self.prompt = prompt
        self.successes = 0  # Positive feedback count
        self.failures = 0   # Negative feedback count
        
        # Beta distribution parameters for Thompson sampling
        # Initialize with weak prior of 1 success, 1 failure
        self.alpha = 1.0  # Prior successes + observed successes
        self.beta = 1.0   # Prior failures + observed failures
    
    def update(self, reward: float) -> None:
        """Update the arm with a new reward observation.
        
        Args:
            reward: Predicted probability of positive feedback (0-1)
        """
        # Convert probabilistic reward to Bernoulli outcome for updating
        # We could also use the raw probability to update, but this approach 
        # maintains the interpretation of alpha/beta as pseudo-counts
        outcome = np.random.binomial(1, reward)
        
        if outcome == 1:
            self.successes += 1
            self.alpha += 1
        else:
            self.failures += 1
            self.beta += 1
    
    def sample(self) -> float:
        """Sample from the arm's posterior distribution.
        
        Returns:
            Sampled estimate of success probability
        """
        return np.random.beta(self.alpha, self.beta)
    
    @property
    def mean_estimate(self) -> float:
        """Get the mean estimate of the arm's success probability.
        
        Returns:
            Mean estimate (alpha / (alpha + beta))
        """
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def total_pulls(self) -> int:
        """Get the total number of times this arm has been pulled.
        
        Returns:
            Total pulls (successes + failures)
        """
        return self.successes + self.failures

class ThompsonBandit:
    """Thompson sampling bandit algorithm for prompt optimization."""
    
    def __init__(self, candidates: List[Prompt]):
        """Initialize the bandit with candidate prompts.
        
        Args:
            candidates: List of candidate prompts to evaluate
        """
        self.arms = [BanditArm(prompt) for prompt in candidates]
    
    def select_arm(self) -> BanditArm:
        """Select an arm to pull using Thompson sampling.
        
        Returns:
            The selected arm
        """
        if not self.arms:
            raise ValueError("No arms available")
        
        # Sample from each arm's posterior
        samples = [arm.sample() for arm in self.arms]
        
        # Select the arm with the highest sample
        best_idx = np.argmax(samples)
        return self.arms[best_idx]
    
    def update(self, arm: BanditArm, reward: float) -> None:
        """Update an arm with a reward observation.
        
        Args:
            arm: The arm to update
            reward: The observed reward (0-1)
        """
        arm.update(reward)
    
    def best_arm(self) -> BanditArm:
        """Get the arm with the highest estimated success probability.
        
        Returns:
            The best arm
        """
        if not self.arms:
            raise ValueError("No arms available")
        
        # Use mean estimates for best arm selection
        means = [arm.mean_estimate for arm in self.arms]
        best_idx = np.argmax(means)
        return self.arms[best_idx]
    
    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all arms.
        
        Returns:
            List of arm statistics
        """
        return [
            {
                "prompt_id": arm.prompt.id,
                "mean_estimate": arm.mean_estimate,
                "total_pulls": arm.total_pulls,
                "successes": arm.successes,
                "failures": arm.failures,
                "alpha": arm.alpha,
                "beta": arm.beta
            }
            for arm in self.arms
        ]