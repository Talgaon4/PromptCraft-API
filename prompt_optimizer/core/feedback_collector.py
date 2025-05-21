# prompt_optimizer/core/feedback_collector.py

from typing import Dict, List, Optional, Any
from prompt_optimizer.storage.base import BaseStorage
from prompt_optimizer.core.models import Feedback, Response, PromptInstance

class FeedbackCollector:
    """Collects and manages feedback on responses."""

    def __init__(
        self,
        feedback_storage: BaseStorage[Feedback],
        response_storage: BaseStorage[Response],
        instance_storage: BaseStorage[PromptInstance]
    ):
        """Initialize with storage implementations."""
        self.feedback_storage = feedback_storage
        self.response_storage = response_storage
        self.instance_storage = instance_storage

    def record_feedback(
        self,
        response_id: str,
        is_positive: bool,
        score: Optional[float] = None,
        comments: Optional[str] = None
    ) -> Feedback:
        """Record feedback for a response."""
        # Verify response exists
        response = self.response_storage.get(response_id)
        if not response:
            raise ValueError(f"Response with ID {response_id} not found")

        # Create and save feedback
        feedback = Feedback(
            response_id=response_id,
            is_positive=is_positive,
            score=score,
            comments=comments
        )
        return self.feedback_storage.save(feedback)

    def get_feedback_for_response(self, response_id: str) -> List[Feedback]:
        """Get all feedback for a specific response."""
        return self.feedback_storage.list({"response_id": response_id})

    def get_feedback_for_prompt(self, prompt_id: str, include_responses: bool = False) -> List[Dict[str, Any]]:
        """Get all feedback for a specific prompt."""
        # Step 1: Get all instances for this prompt
        instances = self.instance_storage.list({"prompt_id": prompt_id})
        
        # Step 2: For each instance, get all responses
        result = []
        for instance in instances:
            responses = self.response_storage.list({"prompt_instance_id": instance.id})
            
            # Step 3: For each response, get all feedback
            for response in responses:
                feedback_items = self.get_feedback_for_response(response.id)
                
                for feedback in feedback_items:
                    feedback_data = {
                        "feedback_id": feedback.id,
                        "response_id": response.id,
                        "prompt_instance_id": instance.id,
                        "prompt_id": prompt_id,
                        "is_positive": feedback.is_positive,
                        "score": feedback.score,
                        "comments": feedback.comments,
                        "created_at": feedback.created_at
                    }
                    
                    if include_responses:
                        feedback_data["response_content"] = response.content
                        feedback_data["formatted_prompt"] = instance.formatted_text
                        
                    result.append(feedback_data)
        
        return result

    def calculate_feedback_stats(self, prompt_id: str) -> Dict[str, Any]:
        """Calculate statistics for feedback on a prompt."""
        feedback_items = self.get_feedback_for_prompt(prompt_id)
        
        if not feedback_items:
            return {
                "prompt_id": prompt_id,
                "total_feedback": 0,
                "positive_ratio": 0,
                "average_score": 0,
                "has_sufficient_data": False,
                "sufficient_threshold": 5
            }
            
        total = len(feedback_items)
        positive_count = sum(1 for item in feedback_items if item["is_positive"])
        scores = [item["score"] for item in feedback_items if item["score"] is not None]
        
        # Calculate actual statistics
        positive_ratio = positive_count / total if total > 0 else 0
        average_score = sum(scores) / len(scores) if scores else None
        
        return {
            "prompt_id": prompt_id,
            "total_feedback": total,
            "positive_ratio": positive_ratio,
            "average_score": average_score,
            "has_sufficient_data": total >= 5,  # Threshold for sufficient data
            "sufficient_threshold": 5
        }