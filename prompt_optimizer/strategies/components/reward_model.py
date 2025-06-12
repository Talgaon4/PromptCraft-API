# prompt_optimizer/strategies/reward_model.py

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import os
from datetime import datetime

class RewardModel:
    """Predicts user satisfaction with responses based on historical feedback."""
    
    def __init__(self, 
                 model_dir: str = "./models",
                 validation_size: float = 0.2,
                 random_state: int = 42):
        """Initialize the reward model.
        
        Args:
            model_dir: Directory to save trained models
            validation_size: Portion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model_dir = model_dir
        self.validation_size = validation_size
        self.random_state = random_state
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def _create_features(self, prompt: str, response: str) -> np.ndarray:
        """Convert prompt and response to feature vector.
        
        In a real implementation, you would use embeddings from a language model.
        This is a simplified placeholder that uses basic text statistics.
        
        Args:
            prompt: The prompt text
            response: The response text
            
        Returns:
            Feature vector representation
        """
        # Simple feature extraction (replace with embeddings in production)
        features = [
            len(prompt),                      # Prompt length
            len(response),                    # Response length
            len(prompt.split()),              # Word count in prompt
            len(response.split()),            # Word count in response
            response.count('?'),              # Question marks in response
            response.count('.'),              # Periods in response
            sum(1 for c in response if c.isupper()) / max(1, len(response)),  # Uppercase ratio
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the reward model on historical feedback.
        
        Args:
            feedback_data: List of dictionaries with keys:
                          'prompt': The prompt text
                          'response': The response text
                          'score': Numeric rating (0-1)
                          
        Returns:
            Training metrics including validation performance
        """
        if not feedback_data:
            raise ValueError("Cannot train on empty feedback data")
        
        # Extract features and labels
        X = []
        y = []
        
        for item in feedback_data:
            prompt = item.get('prompt', '')
            response = item.get('response', '')
            score = item.get('score', 0)

            features = self._create_features(prompt, response)[0]  # Remove batch dimension
            X.append(features)
            y.append(score)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, random_state=self.random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = self.model.predict_proba(X_train)[:, 1]
        val_preds = self.model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_preds) if len(set(y_train)) > 1 else 0.5
        val_auc = roc_auc_score(y_val, val_preds) if len(set(y_val)) > 1 else 0.5
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"reward_model_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.is_trained = True
        
        return {
            "train_auc": train_auc,
            "val_auc": val_auc,
            "n_samples": len(X),
            "positive_rate": y.mean(),
            "model_path": model_path
        }
    
    def predict(self, prompt: str, response: str) -> float:
        """Predict the probability of positive feedback for a prompt-response pair.
        
        Args:
            prompt: The prompt text
            response: The response text
            
        Returns:
            Probability of positive feedback (0-1)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        features = self._create_features(prompt, response)
        pred = self.model.predict_proba(features)[0, 1]  # Probability of positive class
        return float(pred)
    
    def load(self, model_path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True