"""LLM service implementation using OpenAI with configuration support."""

from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
from openai import OpenAI
from prompt_optimizer.config import config

# Load environment variables from .env file
load_dotenv()

class LLMService:
    """Interface for LLM generation using OpenAI with configurable settings."""
    
    def __init__(self, config_instance=None, **overrides):
        """Initialize the LLM service.
        
        Args:
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides
                - api_key: OpenAI API key
                - model: OpenAI model name
                - max_tokens: Maximum tokens to generate
                - temperature: Generation temperature
                - system_prompt: Default system prompt
        """
        # Use provided config or global config
        self.config = config_instance or config
        
        # Apply any direct overrides
        self.api_key = overrides.get('api_key', self.config.OPENAI_API_KEY)
        self.model = overrides.get('model', self.config.LLM_MODEL)
        self.max_tokens = overrides.get('max_tokens', self.config.MAX_TOKENS)
        self.temperature = overrides.get('temperature', self.config.TEMPERATURE)
        self.system_prompt = overrides.get('system_prompt', 
            getattr(self.config, 'SYSTEM_PROMPT', "You are a helpful assistant. Be concise."))
        
        # Validation
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in the .env file or provide it as an argument.")
            
        # Configure the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        print(f"LLMService initialized with model: {self.model}")
        
    def generate(self, prompt: str, temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None) -> str:
        """Generate a response for the given prompt using OpenAI.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (overrides config if provided)
            max_tokens: Maximum tokens to generate (overrides config if provided)
            
        Returns:
            Generated text response
        """
        try:
            # Use provided parameters or fall back to config
            use_temperature = temperature if temperature is not None else self.temperature
            use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=use_temperature,
                max_tokens=use_max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = f"Error generating response from OpenAI: {str(e)}"
            print(error_msg)
            return f"Error generating response: {str(e)}"
    
    def optimize_prompt(self, current_prompt: str, 
                       feedback_examples: List[Dict[str, Any]], 
                       max_tokens: Optional[int] = None) -> str:
        """Optimizes a prompt using a more token-efficient method.
        
        Args:
            current_prompt: The current prompt template
            feedback_examples: List of feedback examples
            max_tokens: Max tokens for optimization response (overrides config)
            
        Returns:
            Optimized prompt text
        """
        # Use provided max_tokens or config value, with a smaller default for optimization
        optimization_max_tokens = max_tokens or min(100, self.max_tokens)
        
        # Create optimization prompt
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
            temperature=0.8,  # More creative for optimization
            max_tokens=optimization_max_tokens
        )
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current LLM settings (useful for debugging)."""
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'system_prompt': self.system_prompt,
            'api_key_set': bool(self.api_key)
        }
    
    def update_settings(self, **new_settings):
        """Update LLM settings at runtime.
        
        Args:
            **new_settings: Settings to update (model, max_tokens, temperature, system_prompt)
        """
        if 'model' in new_settings:
            self.model = new_settings['model']
        if 'max_tokens' in new_settings:
            self.max_tokens = new_settings['max_tokens']
        if 'temperature' in new_settings:
            self.temperature = new_settings['temperature']
        if 'system_prompt' in new_settings:
            self.system_prompt = new_settings['system_prompt']
        
        print(f"LLM settings updated: {new_settings}")