# prompt_optimizer/config.py

"""
Simple configuration for PromptCraft API.
Just import and use: from prompt_optimizer.config import config
"""

import os
from typing import Optional


class Config:
    """Simple configuration class - just change the values below to your preferences"""
    
    # === OPTIMIZATION SETTINGS ===
    # Change these numbers to whatever you want
    OPTIMIZATION_THRESHOLD = 5
    MIN_FEEDBACK_SAMPLES = 10  
    CONFIDENCE_LEVEL = 0.95
    AUTO_APPLY = False
    DEFAULT_STRATEGY = 'simple_ai'
    
    # === LLM SETTINGS ===
    # The API key will try environment variable, others you can change directly
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Only this needs env var
    LLM_MODEL = 'gpt-3.5-turbo'
    MAX_TOKENS = 250
    TEMPERATURE = 0.7
    
    # === STORAGE SETTINGS ===
    DEFAULT_STORAGE_DIR = './data'
    STREAMLIT_STORAGE_DIR = './streamlit_data'
    TEST_STORAGE_DIR = './test_data'
    
    # === INTERFACE SETTINGS ===
    AUTO_CHECK_INTERVAL = 30  # seconds
    AUTO_CHECK_INTERVAL_HOURS = 1  # hours (for production)
    AUTO_CHECK_INTERVAL_SECONDS = 30  # seconds (for demo/testing)
    DEMO_MODE = True
    
    # === STRATEGY SPECIFIC ===
    # Simple AI Strategy
    SIMPLE_AI_MIN_SAMPLES = 5
    SIMPLE_AI_MIN_POSITIVE_RATE = 0.8
    SIMPLE_AI_MAX_EXAMPLES = 5  # Max examples to show AI for optimization
    
    # Reward Model Strategy  
    REWARD_MODEL_MIN_SAMPLES = 30
    REWARD_MODEL_MAX_CANDIDATES = 5
    REWARD_MODEL_MAX_QUERIES = 100  # Max queries per candidate
    REWARD_MODEL_VALIDATION_SIZE = 0.2  # 20% for validation
    REWARD_MODEL_MIN_AUC = 0.6  # Minimum AUC score for model
    
    # === LLM PROMPTS ===
    SYSTEM_PROMPT = "You are a helpful assistant. Be concise."
    
    # === LOGGING SETTINGS ===
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOGS_DIR = "./logs"
    
    # === PERFORMANCE SETTINGS ===
    IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement required
    
    # === OPTIONAL: Environment variable overrides ===
    # Only use these if you want to override for deployment
    # Most of the time you won't need this
    @classmethod
    def _apply_env_overrides(cls):
        """Apply environment variable overrides if they exist"""
        if os.getenv('OPTIMIZATION_THRESHOLD'):
            cls.OPTIMIZATION_THRESHOLD = int(os.getenv('OPTIMIZATION_THRESHOLD'))
        if os.getenv('LLM_MODEL'):
            cls.LLM_MODEL = os.getenv('LLM_MODEL')
        if os.getenv('MAX_TOKENS'):
            cls.MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
        # Add more as needed
    
    @classmethod
    def get_threshold_for_strategy(cls, strategy: str) -> int:
        """Get the appropriate threshold for a strategy"""
        if strategy == 'simple_ai':
            return cls.SIMPLE_AI_MIN_SAMPLES
        elif strategy == 'reward_model_bandit':
            return cls.REWARD_MODEL_MIN_SAMPLES
        else:
            return cls.OPTIMIZATION_THRESHOLD
    
    @classmethod
    def print_config(cls):
        """Print current configuration (useful for debugging)"""
        print("=== PromptCraft Configuration ===")
        print(f"Optimization Threshold: {cls.OPTIMIZATION_THRESHOLD}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Storage Dir: {cls.DEFAULT_STORAGE_DIR}")
        print(f"Strategy: {cls.DEFAULT_STRATEGY}")
        print("=" * 35)


# Create a global config instance
config = Config()


def create_config(**overrides):
    """Create a config instance with optional overrides"""
    # Start with default config
    new_config = Config()
    
    # Apply any overrides
    for key, value in overrides.items():
        key_upper = key.upper()
        if hasattr(new_config, key_upper):
            setattr(new_config, key_upper, value)
        else:
            # Try some common parameter name mappings
            mappings = {
                'optimization_threshold': 'OPTIMIZATION_THRESHOLD',
                'threshold': 'OPTIMIZATION_THRESHOLD', 
                'model': 'LLM_MODEL',
                'api_key': 'OPENAI_API_KEY',
                'max_tokens': 'MAX_TOKENS',
                'temperature': 'TEMPERATURE',
                'storage_dir': 'DEFAULT_STORAGE_DIR',
                'strategy': 'DEFAULT_STRATEGY'
            }
            
            if key in mappings:
                setattr(new_config, mappings[key], value)
            else:
                print(f"Warning: Unknown config parameter '{key}' ignored")
    
    return new_config


# Optional: Helper functions for common patterns
def get_storage_dir(context: str = 'default', config_instance=None) -> str:
    """Get storage directory for specific context"""
    cfg = config_instance or config
    dirs = {
        'default': cfg.DEFAULT_STORAGE_DIR,
        'streamlit': cfg.STREAMLIT_STORAGE_DIR, 
        'test': cfg.TEST_STORAGE_DIR
    }
    return dirs.get(context, cfg.DEFAULT_STORAGE_DIR)


def get_llm_settings(config_instance=None) -> dict:
    """Get LLM settings as a dictionary"""
    cfg = config_instance or config
    return {
        'api_key': cfg.OPENAI_API_KEY,
        'model': cfg.LLM_MODEL,
        'max_tokens': cfg.MAX_TOKENS,
        'temperature': cfg.TEMPERATURE
    }


# For debugging - print config when imported
if __name__ == "__main__":
    config.print_config()