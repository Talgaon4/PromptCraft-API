"""Tracks prompt usage and responses."""

from typing import Dict, List, Optional, Any

from prompt_optimizer.core.models import Prompt, PromptInstance, Response
from prompt_optimizer.storage.base import BaseStorage


class ResponseTracker:
    """Tracks prompt usage and responses."""

    def __init__(
        self,
        prompt_storage: BaseStorage[Prompt],
        instance_storage: BaseStorage[PromptInstance],
        response_storage: BaseStorage[Response]
    ):
        """Initialize with storage implementations."""
        self.prompt_storage = prompt_storage
        self.instance_storage = instance_storage
        self.response_storage = response_storage

    def record_prompt_use(
        self,
        prompt_id: str,
        formatted_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PromptInstance:
        """Record the use of a prompt."""
        # Verify prompt exists
        prompt = self.prompt_storage.get(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt with ID {prompt_id} not found")

        # Create and save instance
        instance = PromptInstance(
            prompt_id=prompt_id,
            formatted_text=formatted_text,
            context=context or {}
        )
        return self.instance_storage.save(instance)

    def record_response(
        self,
        prompt_instance_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Response:
        """Record a response to a prompt instance."""
        # Verify instance exists
        instance = self.instance_storage.get(prompt_instance_id)
        if not instance:
            raise ValueError(f"Prompt instance with ID {prompt_instance_id} not found")

        # Create and save response
        response = Response(
            prompt_instance_id=prompt_instance_id,
            content=content,
            metadata=metadata or {}
        )
        return self.response_storage.save(response)

    def get_responses_for_prompt(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get all responses for a specific prompt."""
        # Get all instances of this prompt
        instances = self.instance_storage.list({"prompt_id": prompt_id})
        
        result = []
        for instance in instances:
            # Get responses for each instance
            responses = self.response_storage.list({"prompt_instance_id": instance.id})
            
            for response in responses:
                result.append({
                    "prompt_id": prompt_id,
                    "instance_id": instance.id,
                    "response_id": response.id,
                    "formatted_text": instance.formatted_text,
                    "response_content": response.content,
                    "context": instance.context,
                    "metadata": response.metadata
                })
                
        return result
