"""LLM service implementation using OpenAI."""

from typing import Dict, Any, Optional, List  # Added List import
import os
from dotenv import load_dotenv
from openai import OpenAI  # Updated import for new OpenAI API

# Load environment variables from .env file
load_dotenv()

class LLMService:
    """Interface for LLM generation using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: The OpenAI model to use.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in the .env file or provide it as an argument.")
            
        # Configure the OpenAI client - updated for newer API
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Token efficiency settings
        self.max_tokens = 250  # Default max tokens to save usage
        self.system_prompt = "You are a helpful assistant. Be concise."
        
        print(f"LLMService initialized with model: {self.model}")
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Generate a response for the given prompt using OpenAI.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness in generation
            max_tokens: Maximum tokens to generate (overrides default)
            
        Returns:
            Generated text response
        """
        try:
            # Use provided max_tokens or default
            tokens_limit = max_tokens or self.max_tokens
            
            # Updated API call format for OpenAI >= 1.0.0
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=tokens_limit
            )
            
            # Extract the response text - updated for newer API response format
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response from OpenAI: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def optimize_prompt(self, current_prompt: str, 
                       feedback_examples: List[Dict[str, Any]], 
                       max_tokens: int = 100) -> str:
        """Optimizes a prompt using a more token-efficient method.
        
        Args:
            current_prompt: The current prompt template
            feedback_examples: List of feedback examples
            max_tokens: Max tokens for optimization response
            
        Returns:
            Optimized prompt text
        """
        # Create a simplified optimization request
        optimization_prompt = f"""Improve this prompt: "{current_prompt}"
        
        Based on user feedback:
        """
        
        # Add at most 3 examples to save tokens
        for i, example in enumerate(feedback_examples[:3]):
            rating = "👍 Positive" if example.get("is_positive", False) else "👎 Negative"
            optimization_prompt += f"\nExample {i+1} ({rating}): {example.get('comments', 'No comment')}"
        
        optimization_prompt += """
        
        Make the prompt more effective but keep it concise.
        Return only the improved prompt text with no explanation or additional text.
        Keep the same {placeholders} if present.
        """
        
        # Use higher temperature for creativity but strict token limit
        return self.generate(
            prompt=optimization_prompt,
            temperature=0.8,
            max_tokens=max_tokens
        )