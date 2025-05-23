#!/usr/bin/env python
"""
Interactive Test Script for PromptCraft API with OpenAI integration.
This script allows you to test prompt optimization with real OpenAI API calls
and collect feedback interactively for any type of prompt.
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import PromptCraft components
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.response_objects import PromptResult, OptimizationResult, ValidationResult, OperationResult
from prompt_optimizer.config import Config, config

# Load environment variables from .env file
load_dotenv()

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Sample test data for different prompt types
SAMPLE_DATA = {
    "summarization": [
        "The European Space Agency's Euclid space telescope has sent back its first full-color images of deep space, and they're stunning. Launched in July, Euclid is designed to map the large-scale structure of the universe and help scientists understand dark matter and dark energy. The first five images showcase Euclid's incredible capabilities, capturing everything from spiral galaxies to massive galaxy clusters with unprecedented clarity. Unlike NASA's James Webb Space Telescope, which takes detailed images of small areas, Euclid can capture vast regions of the sky in a single shot, allowing scientists to study how matter is distributed throughout the cosmos.",
        "Researchers at MIT have developed a new artificial intelligence system that can detect early signs of Alzheimer's disease from brain scans up to five years before clinical symptoms appear. The deep learning model was trained on thousands of MRI scans and can identify subtle patterns that human radiologists often miss. In validation tests, the system achieved 92% accuracy, significantly outperforming current diagnostic methods. This breakthrough could dramatically improve early intervention strategies, potentially slowing disease progression when treatments are most effective. The team plans to conduct clinical trials next year and hopes the technology could be available in hospitals within the next three years.",
        "A groundbreaking study published in Nature Climate Change reveals that global efforts to reduce carbon emissions are finally showing measurable effects. For the first time since the industrial revolution, researchers documented a slight decrease in atmospheric carbon concentration that cannot be attributed to economic downturns or pandemic effects. The study analyzed data from over 300 monitoring stations worldwide and found that increased renewable energy adoption, improved energy efficiency, and successful carbon capture technologies have collectively begun to bend the curve. However, scientists caution that while promising, this represents only the beginning of what needs to be a much steeper decline to meet climate goals set in the Paris Agreement.",
        "Apple unveiled its latest iPhone 15 lineup today at a special event held at its Cupertino headquarters. The new devices feature significant camera upgrades, including a 48-megapixel main sensor and improved low-light performance across all models. The Pro versions introduce a titanium frame, replacing stainless steel, making them lighter while maintaining durability. All models now use USB-C instead of Lightning, bringing Apple in line with EU regulations and industry standards. The company also introduced satellite connectivity for messaging and emergency services in areas without cellular coverage. Pre-orders start this Friday with devices shipping next week."
    ],
    "classification": [
        "I absolutely loved my stay! The staff was incredibly attentive, the room was spotless, and the ocean view was breathtaking. Worth every penny and I'll definitely return next year.",
        "Terrible experience from start to finish. Room wasn't ready at check-in, found hair in the bathroom, and staff was completely unhelpful when we reported issues. Avoid at all costs.",
        "Mixed feelings about this place. Great location and beautiful property, but the service was inconsistent and the prices at the restaurant were outrageous. Might consider staying again if they improve their service.",
        "Pretty average hotel, nothing special but nothing terrible either. Clean rooms, standard amenities, friendly enough staff. Wouldn't go out of my way to stay here again, but it was fine for a business trip."
    ],
    "translation": [
        "La vida es bella, especialmente cuando disfrutamos los pequeños momentos con nuestros seres queridos.",
        "Je voudrais réserver une table pour deux personnes ce soir à vingt heures, s'il vous plaît.",
        "Die Wissenschaft hat viele Fortschritte gemacht, aber es gibt noch viel zu entdecken.",
        "Oggi ho mangiato una pizza deliziosa con i miei amici al ristorante italiano."
    ],
    "creative_writing": [
        "Write a short story about a robot that develops emotions",
        "Create a poem about autumn leaves falling in a city park",
        "Write a dialogue between the sun and the moon",
        "Describe an alien planet with unusual physical properties"
    ],
    "coding": [
        "Write a function to calculate the Fibonacci sequence",
        "Create a simple web scraper to extract headlines from a news website",
        "Implement a binary search algorithm",
        "Write a class to represent a bank account with deposit and withdrawal methods"
    ]
}

# Default prompt templates for different types
DEFAULT_PROMPTS = {
    "summarization": "Summarize the following text in a funny and entertaining way: {input_text}",
    "classification": "Classify the sentiment of this review as 'positive', 'negative', or 'neutral': {input_text}",
    "translation": "Translate the following text into English: {input_text}",
    "creative_writing": "Use the following prompt to create a creative piece: {input_text}",
    "coding": "Provide clean, efficient code for the following task: {input_text}"
}

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

def print_section(text):
    """Print a section title."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(text)}{Colors.ENDC}")

def print_result(result, operation_name):
    """Print a standardized result with proper formatting."""
    if result.success:
        print(f"{Colors.GREEN}✓ {operation_name} successful: {result.message}{Colors.ENDC}")
        if hasattr(result, 'timestamp'):
            print(f"  Time: {result.timestamp}")
    else:
        print(f"{Colors.RED}✗ {operation_name} failed: {result.message}{Colors.ENDC}")
        if result.errors:
            for error in result.errors:
                print(f"  Error: {Colors.RED}{error}{Colors.ENDC}")

def initialize_components():
    """Initialize the PromptOptimizer with OpenAI integration."""
    print_section("Initializing Components")
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"{Colors.RED}Error: OPENAI_API_KEY not found in environment.{Colors.ENDC}")
        print("Please create a .env file with your OpenAI API key.")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Found OpenAI API key{Colors.ENDC}")
    
    # Create LLM service
    try:
        llm_service = LLMService(api_key=api_key, model="gpt-3.5-turbo")
        print(f"{Colors.GREEN}✓ LLM service initialized{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize LLM service: {str(e)}{Colors.ENDC}")
        sys.exit(1)
    
    # Configure the optimizer with SimpleAI strategy
    try:
        # Explicitly pass the API key in configuration
        optimizer = PromptOptimizer(
            storage_dir="./test_data",
            optimization_threshold=3,  # Lower threshold for testing
            DEFAULT_STRATEGY="simple_ai",  # Use correct parameter name
            OPENAI_API_KEY=api_key  # Explicitly pass API key
        )
        
        # Inject the LLM service into the strategy
        optimizer.optimizer.strategy.llm_service = llm_service
        
        # Set slightly higher minimum positive rate to trigger optimization sooner
        optimizer.optimizer.strategy.min_positive_rate = 0.7
        
        print(f"{Colors.GREEN}✓ Optimizer initialized with OpenAI integration{Colors.ENDC}")
        return optimizer, llm_service
        
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize optimizer: {str(e)}{Colors.ENDC}")
        sys.exit(1)

def select_prompt_type():
    """Let the user select a prompt type to test."""
    print_section("Select Prompt Type")
    
    print("Available prompt types:")
    for i, prompt_type in enumerate(DEFAULT_PROMPTS.keys(), 1):
        print(f"{i}. {prompt_type.capitalize()}")
    
    while True:
        try:
            choice = int(input("\nEnter number (1-5): "))
            if 1 <= choice <= len(DEFAULT_PROMPTS):
                prompt_type = list(DEFAULT_PROMPTS.keys())[choice-1]
                return prompt_type
            else:
                print(f"{Colors.YELLOW}Please enter a number between 1 and {len(DEFAULT_PROMPTS)}{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.YELLOW}Please enter a valid number{Colors.ENDC}")

def create_prompt(optimizer, prompt_type):
    """Create a new prompt for testing."""
    print_section(f"Creating {prompt_type.capitalize()} Prompt")
    
    # Get default template for selected type
    default_template = DEFAULT_PROMPTS[prompt_type]
    
    # Get user input or use default
    print("Enter prompt template (or press Enter for default):")
    print(f"Default: {default_template}")
    prompt_text = input().strip()
    
    if not prompt_text:
        prompt_text = default_template
        print(f"Using default: {prompt_text}")
    
    print("Enter description (optional):")
    description = input().strip() or f"{prompt_type.capitalize()} prompt"
    
    # Create the prompt using new API
    result = optimizer.register_prompt(text=prompt_text, description=description)
    
    if result.success:
        print_result(result, "Prompt creation")
        print(f"  Prompt ID: {Colors.CYAN}{result.prompt_id}{Colors.ENDC}")
        
        # Detect the placeholder format in the prompt
        import re
        placeholders = re.findall(r'\{([^{}]+)\}', prompt_text)
        placeholder = placeholders[0] if placeholders else "input_text"
        
        return result.prompt_id, result.prompt_text or prompt_text, placeholder
    else:
        print_result(result, "Prompt creation")
        sys.exit(1)

def update_prompt(optimizer, prompt_id, new_text, description=None):
    """Safely update a prompt with a new template.
    
    Args:
        optimizer: PromptOptimizer instance
        prompt_id: ID of the prompt to update
        new_text: New prompt template text
        description: Optional description update
        
    Returns:
        Tuple of (success, new_prompt_id, error_message)
    """
    try:
        # First get the current prompt to preserve any data we don't want to change
        prompt_result = optimizer.get_prompt(prompt_id)
        
        if not prompt_result.success:
            return False, None, prompt_result.message
        
        # Use the optimizer.optimizer.prompt_manager directly
        # since the API doesn't directly expose update_prompt
        update_result = optimizer.optimizer.prompt_manager.update_prompt(
            prompt_id=prompt_id,
            text=new_text,
            description=description if description else prompt_result.data.get('description')
        )
        
        # Return the new prompt ID
        return True, update_result.id, None
        
    except Exception as e:
        return False, None, str(e)

def interactive_testing(optimizer, llm_service, prompt_id, prompt_text, prompt_type, placeholder):
    """Run interactive testing and feedback collection."""
    print_section(f"Interactive Testing: {prompt_type.capitalize()}")
    print(f"Using prompt template: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
    print(f"Input placeholder: {placeholder}")
    print(f"\n{Colors.YELLOW}Commands:{Colors.ENDC}")
    print("- Enter your input text")
    print("- Type 'sample' to use a sample input for this prompt type")
    print("- Type 'optimize' to check optimization readiness and optimize if ready")
    print("- Type 'edit' to manually edit the prompt template")  # New command
    print("- Type 'exit' to end testing")
    
    feedback_count = 0
    sample_index = 0
    
    while True:
        print(f"\n{Colors.BOLD}Enter input text (or command):{Colors.ENDC}")
        user_input = input().strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'sample':
            # Use one of the sample inputs for this prompt type
            if prompt_type in SAMPLE_DATA and SAMPLE_DATA[prompt_type]:
                user_input = SAMPLE_DATA[prompt_type][sample_index % len(SAMPLE_DATA[prompt_type])]
                sample_index += 1
                print(f"Using sample input #{sample_index}:")
                print(f"{Colors.CYAN}{user_input[:150]}{'...' if len(user_input) > 150 else ''}{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}No sample data available for {prompt_type}{Colors.ENDC}")
                continue
        
        # New 'edit' command to manually edit the prompt
        if user_input.lower() == 'edit':
            print_section("Edit Prompt Template")
            print(f"Current template: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
            print(f"\n{Colors.YELLOW}Enter new prompt template (include {{{placeholder}}} placeholder):{Colors.ENDC}")
            new_template = input().strip()
            
            if not new_template:
                print(f"{Colors.YELLOW}No changes made.{Colors.ENDC}")
                continue
                
            if f"{{{placeholder}}}" not in new_template:
                print(f"{Colors.RED}Error: New template must include {{{placeholder}}} placeholder.{Colors.ENDC}")
                continue
                
            # Update the prompt in the system
            success, new_prompt_id, error = update_prompt(
                optimizer, 
                prompt_id, 
                new_template, 
                f"Manually edited {prompt_type} prompt"
            )
            
            if success:
                # Update our tracking variables
                prompt_id = new_prompt_id
                prompt_text = new_template
                
                print(f"{Colors.GREEN}✓ Prompt updated successfully.{Colors.ENDC}")
                print(f"New prompt ID: {Colors.CYAN}{new_prompt_id}{Colors.ENDC}")
                print(f"New template: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Error updating prompt: {error}{Colors.ENDC}")
            
            continue
        
        if user_input.lower() == 'optimize':
            # Handle optimization safely
            try:
                result = check_and_optimize(optimizer, prompt_id, prompt_text, force=True)
                
                # Safely unpack the result
                if result and isinstance(result, tuple) and len(result) == 2:
                    new_id, new_text = result
                    if new_id and new_text:  # Make sure we got valid data
                        prompt_id = new_id
                        prompt_text = new_text
                        print(f"{Colors.GREEN}Now using the optimized prompt:{Colors.ENDC}")
                        print(f"{Colors.CYAN}{prompt_text}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}Error during optimization: {str(e)}{Colors.ENDC}")
                print("Continuing with the existing prompt.")
            continue
            
        if not user_input or user_input.lower() in ['sample', 'optimize', 'edit', 'exit']:
            continue
        
        # Format the prompt with the detected placeholder
        formatted_prompt = prompt_text.replace(f"{{{placeholder}}}", user_input)
        
        # Record prompt usage using new API
        print(f"{Colors.YELLOW}Recording prompt usage...{Colors.ENDC}")
        usage_result = optimizer.record_prompt_use(
            prompt_id=prompt_id,
            formatted_text=formatted_prompt
        )
        
        if not usage_result.success:
            print_result(usage_result, "Prompt usage recording")
            continue
        
        instance_id = usage_result.data['instance_id']
        print(f"{Colors.GREEN}✓ Recorded usage (ID: {instance_id}){Colors.ENDC}")
        
        # Generate response using OpenAI
        print(f"{Colors.YELLOW}Generating response...{Colors.ENDC}")
        try:
            response = llm_service.generate(formatted_prompt)
            print(f"\n{Colors.GREEN}Generated response:{Colors.ENDC}")
            print(response)
            
            # Record the response using new API
            response_result = optimizer.record_response(
                prompt_instance_id=instance_id,
                content=response
            )
            
            if not response_result.success:
                print_result(response_result, "Response recording")
                continue
            
            response_id = response_result.data['response_id']
            print(f"{Colors.GREEN}✓ Recorded response (ID: {response_id}){Colors.ENDC}")
            
            # Collect feedback - customize based on prompt type
            print(f"\n{Colors.BOLD}Was this a good response? (y/n):{Colors.ENDC}")
            is_positive = input().lower().startswith('y')
            
            print(f"{Colors.BOLD}Score (0-10):{Colors.ENDC}")
            try:
                score = float(input() or "5") / 10.0  # Convert to 0-1 scale, default to 5
            except ValueError:
                score = 0.5  # Default
            
            print(f"{Colors.BOLD}Comments (optional):{Colors.ENDC}")
            comments = input().strip()
            
            # Record feedback using new API
            feedback_result = optimizer.record_feedback(
                response_id=response_id,
                is_positive=is_positive,
                score=score,
                comments=comments if comments else None
            )
            
            if feedback_result.success:
                feedback_count += 1
                print(f"{Colors.GREEN}✓ Feedback recorded ({feedback_count}) - ID: {feedback_result.data['feedback_id']}{Colors.ENDC}")
                
                # Check optimization readiness every 3 feedback items
                if feedback_count % 3 == 0:
                    try:
                        check_and_optimize(optimizer, prompt_id, prompt_text)
                    except Exception as e:
                        print(f"{Colors.RED}Error checking optimization: {str(e)}{Colors.ENDC}")
            else:
                print_result(feedback_result, "Feedback recording")
                    
        except Exception as e:
            print(f"{Colors.RED}Error processing input: {str(e)}{Colors.ENDC}")

def safe_get_prompt(optimizer, prompt_id):
    """Safely get a prompt with error handling."""
    try:
        result = optimizer.get_prompt(prompt_id)
        if result.success:
            return result.data, None
        else:
            return None, result.message
    except Exception as e:
        print(f"{Colors.RED}Error retrieving prompt {prompt_id}: {str(e)}{Colors.ENDC}")
        return None, str(e)

def check_and_optimize(optimizer, prompt_id, prompt_text, force=False) -> Tuple[Optional[str], str]:
    """Check if a prompt is ready for optimization and optimize if ready."""
    print_section("Checking Optimization Readiness")
    
    # Get readiness data using new API
    stats_result = optimizer.get_optimization_stats(prompt_id)
    
    if not stats_result.success:
        print_result(stats_result, "Optimization stats check")
        return None, prompt_text
    
    readiness = stats_result.data
    print(f"Feedback count: {readiness.get('feedback_count', 0)}")
    print(f"Threshold: {readiness.get('threshold', 3)}")
    print(f"Is ready: {readiness.get('is_ready', False)}")
    
    if "strategy_assessment" in readiness:
        print("\nStrategy Assessment:")
        for key, value in readiness["strategy_assessment"].items():
            print(f"- {key}: {value}")
    
    # Try to optimize if ready or forced
    if readiness.get("is_ready", False) or force:
        print_section("Optimizing Prompt")
        
        print(f"Original prompt: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Generating optimization...{Colors.ENDC}")
        
        # Try to optimize the prompt using new API
        opt_result = optimizer.optimize_prompt(prompt_id, force=force)
        
        if opt_result.success:
            print_result(opt_result, "Optimization")
            
            if opt_result.optimization_applied:
                # Optimization was applied, get the new prompt
                new_prompt_data, error = safe_get_prompt(optimizer, opt_result.new_prompt_id)
                
                if new_prompt_data:
                    print(f"Optimized prompt: {Colors.CYAN}{new_prompt_data['text']}{Colors.ENDC}")
                    return opt_result.new_prompt_id, new_prompt_data['text']
                else:
                    print(f"{Colors.YELLOW}Could not retrieve optimized prompt: {error}{Colors.ENDC}")
                    return prompt_id, prompt_text
            else:
                # Optimization was generated but not applied
                if opt_result.data and 'optimized_text' in opt_result.data:
                    optimized_text = opt_result.data['optimized_text']
                    print(f"Generated optimization: {Colors.CYAN}{optimized_text}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Note: Optimization not auto-applied{Colors.ENDC}")
                    return prompt_id, optimized_text  # Return the optimized text
                else:
                    print(f"{Colors.YELLOW}Optimization generated but text not available{Colors.ENDC}")
                    return prompt_id, prompt_text
        else:
            print_result(opt_result, "Optimization")
            
            # Show readiness info if available
            if opt_result.readiness_info:
                print(f"{Colors.YELLOW}Readiness details:{Colors.ENDC}")
                readiness_info = opt_result.readiness_info
                needed = readiness_info.get('threshold', 3) - readiness_info.get('feedback_count', 0)
                if needed > 0:
                    print(f"  Need {needed} more feedback items")
    else:
        print(f"{Colors.YELLOW}Not ready for optimization yet.{Colors.ENDC}")
        needed = readiness.get('threshold', 3) - readiness.get('feedback_count', 0)
        print(f"Need {needed} more feedback items.")
    
    return None, prompt_text

def automatic_demo(optimizer, llm_service, prompt_id, prompt_text, prompt_type, placeholder):
    """Run an automatic demo with pre-defined examples."""
    print_section(f"Automatic {prompt_type.capitalize()} Optimization Demo")
    print("This demo will automatically:")
    print(f"1. Test the prompt with sample {prompt_type} inputs")
    print("2. Record feedback on the responses")
    print("3. Optimize the prompt when ready")
    
    # Ask for confirmation
    print(f"\n{Colors.YELLOW}Run automatic demo? (y/n):{Colors.ENDC}")
    if not input().lower().startswith('y'):
        return
    
    # Check if we have samples for this prompt type
    if prompt_type not in SAMPLE_DATA or not SAMPLE_DATA[prompt_type]:
        print(f"{Colors.RED}No sample data available for {prompt_type}{Colors.ENDC}")
        return
        
    # Use sample data for this prompt type
    sample_inputs = SAMPLE_DATA[prompt_type]
    
    # Determine expected feedback by prompt type
    if prompt_type == "classification":
        # For classification, we'll check if "positive" is in the response for positive reviews
        expected_results = [
            {"expected": "positive", "keywords": ["positive"]},
            {"expected": "negative", "keywords": ["negative"]},
            {"expected": "neutral", "keywords": ["neutral", "mixed"]},
            {"expected": "neutral", "keywords": ["neutral"]}
        ]
    else:
        # For other types, we'll just alternate positive/negative feedback
        expected_results = [{"expected": "good"} for _ in range(len(sample_inputs))]
    
    # Test inputs and collect feedback
    print("\nTesting with sample inputs and collecting feedback...")
    for i, input_text in enumerate(sample_inputs):
        print(f"\n{Colors.BOLD}Sample {i+1}/{len(sample_inputs)}:{Colors.ENDC}")
        print(f"{Colors.CYAN}Input: {input_text[:150]}{'...' if len(input_text) > 150 else ''}{Colors.ENDC}")
        
        # Format the prompt
        formatted_prompt = prompt_text.replace(f"{{{placeholder}}}", input_text)
        
        # Record prompt usage using new API
        usage_result = optimizer.record_prompt_use(
            prompt_id=prompt_id,
            formatted_text=formatted_prompt
        )
        
        if not usage_result.success:
            print_result(usage_result, f"Usage recording for sample {i+1}")
            continue
        
        instance_id = usage_result.data['instance_id']
        
        # Generate response
        print(f"{Colors.YELLOW}Generating response...{Colors.ENDC}")
        try:
            response = llm_service.generate(formatted_prompt)
            print(f"{Colors.GREEN}Response: {response[:150]}{'...' if len(response) > 150 else ''}{Colors.ENDC}")
            
            # Record the response using new API
            response_result = optimizer.record_response(
                prompt_instance_id=instance_id,
                content=response
            )
            
            if not response_result.success:
                print_result(response_result, f"Response recording for sample {i+1}")
                continue
            
            response_id = response_result.data['response_id']
            
            # Determine feedback based on prompt type
            if prompt_type == "classification":
                # Check if response contains expected keywords
                expected = expected_results[i % len(expected_results)]
                is_correct = any(keyword.lower() in response.lower() for keyword in expected["keywords"])
            else:
                # For other types, let's alternate feedback to ensure variety
                is_correct = i % 2 == 0  # Alternate between positive and negative
            
            # Record feedback using new API
            feedback_result = optimizer.record_feedback(
                response_id=response_id,
                is_positive=is_correct,
                score=0.9 if is_correct else 0.2,
                comments=f"{'Good' if is_correct else 'Needs improvement'} response"
            )
            
            if feedback_result.success:
                status_color = Colors.GREEN if is_correct else Colors.RED
                print(f"{status_color}✓ Recorded feedback: {'Positive' if is_correct else 'Negative'}{Colors.ENDC}")
            else:
                print_result(feedback_result, f"Feedback recording for sample {i+1}")
            
            time.sleep(1)  # Pause between requests
            
        except Exception as e:
            print(f"{Colors.RED}Error processing sample {i+1}: {str(e)}{Colors.ENDC}")
    
    # Check optimization readiness and try to optimize
    try:
        result = check_and_optimize(optimizer, prompt_id, prompt_text, force=True)
        
        # If optimization was successful
        if result and isinstance(result, tuple) and len(result) == 2:
            new_prompt_id, optimized_text = result
            if new_prompt_id and optimized_text and optimized_text != prompt_text:
                # Test the optimized prompt with a new input
                print(f"\n{Colors.BOLD}Testing optimized prompt with new input:{Colors.ENDC}")
                
                # Create a new test input based on prompt type
                if prompt_type == "summarization":
                    test_input = "The latest global economic forum concluded with a joint commitment to reducing carbon emissions by 30% before 2030. Leaders from 45 countries signed the agreement, which includes substantial penalties for non-compliance. The deal is being hailed as a breakthrough in climate negotiations, though critics point out that previous agreements have failed to meet targets."
                elif prompt_type == "classification":
                    test_input = "I had a decent experience. The room was clean but small, and the staff was professional but not particularly friendly. The location was convenient though."
                elif prompt_type == "translation":
                    test_input = "Domani andrò al mercato per comprare frutta e verdura fresche."
                elif prompt_type == "creative_writing":
                    test_input = "Write a short story about time travel to ancient Egypt"
                else:
                    test_input = "Create a function that checks if a string is a palindrome"
                
                print(f"\n{Colors.CYAN}Test input: {test_input}{Colors.ENDC}")
                
                try:
                    # Format with optimized prompt
                    formatted_optimized = optimized_text.replace(f"{{{placeholder}}}", test_input)
                    print(f"{Colors.YELLOW}Generating response with optimized prompt...{Colors.ENDC}")
                    optimized_result = llm_service.generate(formatted_optimized)
                    print(f"{Colors.GREEN}Response from optimized prompt: {optimized_result}{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.RED}Error testing optimized prompt: {str(e)}{Colors.ENDC}")
                        
    except Exception as e:
        print(f"{Colors.RED}Error during optimization phase: {str(e)}{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}Demo complete!{Colors.ENDC}")

def main():
    """Main function for the test script."""
    print_header("PromptCraft API OpenAI Integration Test")
    
    try:
        # Initialize components
        optimizer, llm_service = initialize_components()
        
        # Select prompt type
        prompt_type = select_prompt_type()
        
        # Create a test prompt
        prompt_id, prompt_text, placeholder = create_prompt(optimizer, prompt_type)
        
        # Choose mode
        print_section("Testing Mode")
        print("1. Interactive testing (provide feedback on each response)")
        print("2. Automatic demo (using sample inputs with automatic feedback)")
        
        mode = input("\nSelect mode (1/2): ").strip()
        
        if mode == "2":
            automatic_demo(optimizer, llm_service, prompt_id, prompt_text, prompt_type, placeholder)
        else:
            interactive_testing(optimizer, llm_service, prompt_id, prompt_text, prompt_type, placeholder)
        
        print_header("Test Complete")
        print("Thank you for testing PromptCraft API!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\n{Colors.RED}Error during test: {str(e)}{Colors.ENDC}")
        raise

if __name__ == "__main__":
    main()