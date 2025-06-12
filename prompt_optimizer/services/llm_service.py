"""LLM service implementation using OpenAI."""

from typing import Dict, Any, Optional, List
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from prompt_optimizer.config import config
from prompt_optimizer.exceptions import LLMError, ValidationError, validate_not_empty

# Load environment variables from .env file
load_dotenv()

# Simple logging setup
logger = logging.getLogger(__name__)


class LLMService:
    """Interface for LLM generation using OpenAI with proper error handling."""
    
    def __init__(self, config_instance=None, **overrides):
        """Initialize the LLM service.
        
        Args:
            config_instance: Custom config instance (optional)
            **overrides: Direct parameter overrides
            
        Raises:
            LLMError: If initialization fails
            ValidationError: If API key is missing
        """
        try:
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
                raise ValidationError("OpenAI API key is required. Set OPENAI_API_KEY in environment or provide as parameter.")
            
            # Configure the OpenAI client
            self.client = OpenAI(api_key=self.api_key)
            
            logger.info(f"LLMService initialized with model: {self.model}")
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise LLMError(f"LLM initialization failed: {str(e)}") from e
        
    def generate(self, prompt: str, temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None) -> str:
        """Generate a response for the given prompt using OpenAI.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (overrides config if provided)
            max_tokens: Maximum tokens to generate (overrides config if provided)
            
        Returns:
            Generated text response
            
        Raises:
            ValidationError: If prompt is empty
            LLMError: If generation fails
        """
        try:
            # Simple validation
            validate_not_empty(prompt, "Prompt")
            
            # Use provided parameters or fall back to config
            use_temperature = temperature if temperature is not None else self.temperature
            use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Validate parameters
            if not 0 <= use_temperature <= 2:
                raise ValidationError("Temperature must be between 0 and 2")
            if use_max_tokens <= 0:
                raise ValidationError("Max tokens must be positive")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=use_temperature,
                max_tokens=use_max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            logger.debug(f"Generated response of {len(result)} characters")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}") from e
    
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
            
        Raises:
            ValidationError: If inputs are invalid
            LLMError: If optimization fails
        """
        try:
            # Simple validation
            validate_not_empty(current_prompt, "Current prompt")
            
            if not feedback_examples:
                raise ValidationError("Feedback examples cannot be empty")
            
            # Use provided max_tokens or config value, with a smaller default for optimization
            optimization_max_tokens = max_tokens or min(100, self.max_tokens)
            
            # Create optimization prompt
            optimization_prompt = f"""Improve this prompt: "{current_prompt}"
            
            Based on user feedback:
            """
            
            # Add at most 3 examples to save tokens
            for i, example in enumerate(feedback_examples[:3]):
                rating = "👍 High" if (example.get("score") or 0) >= 0.5 else "👎 Low"
                optimization_prompt += f"\nExample {i+1} ({rating})"
            
            optimization_prompt += """
            
            Make the prompt more effective but keep it concise.
            Return only the improved prompt text with no explanation or additional text.
            Keep the same {placeholders} if present.
            """
            
            # Use higher temperature for creativity but strict token limit
            result = self.generate(
                prompt=optimization_prompt,
                temperature=0.8,  # More creative for optimization
                max_tokens=optimization_max_tokens
            )
            
            logger.info("Successfully optimized prompt")
            return result
            
        except (ValidationError, LLMError):
            raise
        except Exception as e:
            logger.error(f"Prompt optimization failed: {str(e)}")
            raise LLMError(f"Failed to optimize prompt: {str(e)}") from e
    
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
            
        Raises:
            ValidationError: If settings are invalid
        """
        try:
            # Validate new settings
            if 'temperature' in new_settings:
                temp = new_settings['temperature']
                if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
                    raise ValidationError("Temperature must be a number between 0 and 2")
                    
            if 'max_tokens' in new_settings:
                tokens = new_settings['max_tokens']
                if not isinstance(tokens, int) or tokens <= 0:
                    raise ValidationError("Max tokens must be a positive integer")
            
            # Apply valid settings
            if 'model' in new_settings:
                self.model = new_settings['model']
            if 'max_tokens' in new_settings:
                self.max_tokens = new_settings['max_tokens']
            if 'temperature' in new_settings:
                self.temperature = new_settings['temperature']
            if 'system_prompt' in new_settings:
                self.system_prompt = new_settings['system_prompt']
            
            logger.info(f"LLM settings updated: {list(new_settings.keys())}")
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to update LLM settings: {str(e)}")
            raise LLMError(f"Failed to update settings: {str(e)}") from e