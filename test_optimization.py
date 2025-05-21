#!/usr/bin/env python
"""
Interactive Test Script for PromptCraft API with OpenAI integration.
This script allows you to test prompt optimization with real OpenAI API calls
and collect feedback interactively.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import PromptCraft components
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.services.llm_service import LLMService
from prompt_optimizer.config import OptimizerConfig

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

# Test hotel reviews for demo purposes
DEFAULT_TEST_REVIEWS = [
    "The room was immaculate and the staff went above and beyond to help us.",
    "Dirty sheets and rude staff. Never again.",
    "Great location, friendly staff, clean rooms. Would definitely stay again!",
    "The room smelled like cigarettes despite being non-smoking.",
    "Fantastic value for money. Modern rooms and helpful concierge.",
    "Paper-thin walls meant we heard everything from neighboring rooms all night.",
    "Clean room but extremely slow check-in process.",
    "Excellent stay!",
    "Terrible experience."
]

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

def print_section(text):
    """Print a section title."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(text)}{Colors.ENDC}")

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
    llm_service = LLMService(api_key=api_key, model="gpt-3.5-turbo")
    print(f"{Colors.GREEN}✓ LLM service initialized{Colors.ENDC}")
    
    # Configure the optimizer with SimpleAI strategy
    optimizer = PromptOptimizer(
        storage_dir="./test_data",
        optimization_threshold=3,  # Lower threshold for testing
        strategy_name="simple_ai"
    )
    
    # Inject the LLM service into the strategy
    optimizer.optimizer.strategy.llm_service = llm_service
    
    # Set slightly higher minimum positive rate to trigger optimization sooner
    optimizer.optimizer.strategy.min_positive_rate = 0.7
    
    print(f"{Colors.GREEN}✓ Optimizer initialized with OpenAI integration{Colors.ENDC}")
    return optimizer, llm_service

def create_prompt(optimizer):
    """Create a new prompt for testing."""
    print_section("Creating Test Prompt")
    
    # Get user input or use default
    print("Enter prompt template (or press Enter for default):")
    prompt_text = input().strip()
    
    if not prompt_text:
        # Better default prompt for hotel review classification
        prompt_text = "Classify the following hotel review as either 'good' or 'bad'. Return only the word 'good' or 'bad': {input_text}"
        print(f"Using default: {prompt_text}")
    
    print("Enter description (optional):")
    description = input().strip() or "Hotel review classifier"
    
    # Create the prompt
    prompt_id = optimizer.register_prompt(text=prompt_text, description=description)
    
    print(f"{Colors.GREEN}✓ Created prompt with ID: {prompt_id}{Colors.ENDC}")
    return prompt_id, prompt_text

def interactive_testing(optimizer, llm_service, prompt_id, prompt_text):
    """Run interactive testing and feedback collection."""
    print_section("Interactive Testing")
    print(f"Using prompt template: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
    print(f"\n{Colors.YELLOW}Commands:{Colors.ENDC}")
    print("- Enter a hotel review to classify")
    print("- Type 'sample' to use a sample review")
    print("- Type 'optimize' to check optimization readiness and optimize if ready")
    print("- Type 'exit' to end testing")
    
    feedback_count = 0
    sample_index = 0
    
    while True:
        print(f"\n{Colors.BOLD}Enter hotel review (or command):{Colors.ENDC}")
        user_input = input().strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'sample':
            # Use one of the sample reviews
            user_input = DEFAULT_TEST_REVIEWS[sample_index % len(DEFAULT_TEST_REVIEWS)]
            sample_index += 1
            print(f"Using sample review #{sample_index}:")
            print(f"{Colors.CYAN}{user_input}{Colors.ENDC}")
        
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
            
        if not user_input or user_input.lower() in ['sample', 'optimize', 'exit']:
            continue
        
        # Format the prompt
        formatted_prompt = prompt_text.replace("{input_text}", user_input)
        
        # Record prompt usage
        print(f"{Colors.YELLOW}Generating response...{Colors.ENDC}")
        try:
            instance_id = optimizer.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=formatted_prompt
            )
            
            # Generate response using OpenAI
            response = llm_service.generate(formatted_prompt)
            print(f"\n{Colors.GREEN}Generated response:{Colors.ENDC}")
            print(response)
            
            # Record the response
            response_id = optimizer.record_response(
                prompt_instance_id=instance_id,
                content=response
            )
            
            # Collect feedback
            print(f"\n{Colors.BOLD}Was this response correct? (y/n):{Colors.ENDC}")
            is_positive = input().lower().startswith('y')
            
            print(f"{Colors.BOLD}Score (0-10):{Colors.ENDC}")
            try:
                score = float(input() or "5") / 10.0  # Convert to 0-1 scale, default to 5
            except ValueError:
                score = 0.5  # Default
            
            print(f"{Colors.BOLD}Comments (optional):{Colors.ENDC}")
            comments = input().strip()
            
            # Record feedback
            feedback_id = optimizer.record_feedback(
                response_id=response_id,
                is_positive=is_positive,
                score=score,
                comments=comments
            )
            
            feedback_count += 1
            print(f"{Colors.GREEN}✓ Feedback recorded ({feedback_count}){Colors.ENDC}")
            
            # Check optimization readiness every 3 feedback items
            if feedback_count % 3 == 0:
                try:
                    check_and_optimize(optimizer, prompt_id, prompt_text)
                except Exception as e:
                    print(f"{Colors.RED}Error checking optimization: {str(e)}{Colors.ENDC}")
                    
        except Exception as e:
            print(f"{Colors.RED}Error processing input: {str(e)}{Colors.ENDC}")

def safe_get_prompt(optimizer, prompt_id):
    """Safely get a prompt with error handling."""
    try:
        prompt = optimizer.get_prompt(prompt_id)
        return prompt
    except Exception as e:
        print(f"{Colors.RED}Error retrieving prompt {prompt_id}: {str(e)}{Colors.ENDC}")
        return None

def check_and_optimize(optimizer, prompt_id, prompt_text, force=False) -> Tuple[Optional[str], str]:
    """Check if a prompt is ready for optimization and optimize if ready."""
    print_section("Checking Optimization Readiness")
    
    # Get readiness data
    try:
        readiness = optimizer.get_optimization_stats(prompt_id)
        print(f"Feedback count: {readiness.get('feedback_count', 0)}")
        print(f"Threshold: {readiness.get('threshold', 3)}")
        print(f"Is ready: {readiness.get('is_ready', False)}")
        
        if "strategy_assessment" in readiness:
            print("\nStrategy Assessment:")
            for key, value in readiness["strategy_assessment"].items():
                print(f"- {key}: {value}")
    except Exception as e:
        print(f"{Colors.RED}Error checking optimization readiness: {str(e)}{Colors.ENDC}")
        return None, prompt_text
    
    # Try to optimize if ready or forced
    if readiness.get("is_ready", False) or force:
        print_section("Optimizing Prompt")
        
        print(f"Original prompt: {Colors.CYAN}{prompt_text}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Generating optimization...{Colors.ENDC}")
        
        try:
            # Try to optimize the prompt
            optimization_result = optimizer.optimize_prompt(prompt_id, force=force)
            
            if optimization_result:
                # Check if the result is a string (direct prompt text)
                if isinstance(optimization_result, str):
                    print(f"{Colors.GREEN}Optimization successful!{Colors.ENDC}")
                    print(f"Optimized prompt: {Colors.CYAN}{optimization_result}{Colors.ENDC}")
                    # Return the same prompt ID with the new text
                    return prompt_id, optimization_result
                
                # If not a string, assume it's a prompt ID
                new_prompt = safe_get_prompt(optimizer, optimization_result)
                
                if new_prompt and isinstance(new_prompt, dict) and 'text' in new_prompt:
                    print(f"{Colors.GREEN}Optimization successful!{Colors.ENDC}")
                    print(f"Optimized prompt: {Colors.CYAN}{new_prompt['text']}{Colors.ENDC}")
                    return optimization_result, new_prompt['text']
                
                # Try to get the parent prompt as a fallback
                print(f"{Colors.YELLOW}Could not retrieve the new prompt. Checking history...{Colors.ENDC}")
                try:
                    # Try to get the prompt history
                    history = optimizer.optimizer.get_optimization_history(prompt_id)
                    if history and len(history) > 0:
                        latest = history[0]
                        print(f"{Colors.GREEN}Found latest version in history!{Colors.ENDC}")
                        print(f"Optimized prompt: {Colors.CYAN}{latest.get('text', prompt_text)}{Colors.ENDC}")
                        return latest.get('prompt_id', prompt_id), latest.get('text', prompt_text)
                except:
                    pass
                    
                print(f"{Colors.YELLOW}Using original prompt as fallback.{Colors.ENDC}")
                return prompt_id, prompt_text
            else:
                print(f"{Colors.YELLOW}Optimization not performed.{Colors.ENDC}")
                print("Possible reasons:")
                print("- Feedback quality doesn't warrant optimization")
                print("- Strategy assessment determined optimization not needed")
        except Exception as e:
            print(f"{Colors.RED}Error during optimization: {str(e)}{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}Not ready for optimization yet.{Colors.ENDC}")
        print(f"Need {readiness.get('threshold', 3) - readiness.get('feedback_count', 0)} more feedback items.")
    
    return None, prompt_text

def automatic_demo(optimizer, llm_service, prompt_id, prompt_text):
    """Run an automatic demo with pre-defined examples."""
    print_section("Automatic Optimization Demo")
    print("This demo will automatically:")
    print("1. Test the prompt with sample hotel reviews")
    print("2. Record feedback based on expected classifications")
    print("3. Optimize the prompt when ready")
    
    # Ask for confirmation
    print(f"\n{Colors.YELLOW}Run automatic demo? (y/n):{Colors.ENDC}")
    if not input().lower().startswith('y'):
        return
    
    # Sample reviews with expected classifications
    review_data = [
        {"text": "The room was immaculate and staff was amazing.", "expected": "good"},
        {"text": "Dirty bathroom and very noisy.", "expected": "bad"},
        {"text": "Great breakfast and comfortable beds.", "expected": "good"},
        {"text": "Staff was rude and check-in took forever.", "expected": "bad"},
        {"text": "Beautiful view and excellent service.", "expected": "good"}
    ]
    
    # Test reviews and collect feedback
    print("\nTesting reviews and collecting feedback...")
    for i, data in enumerate(review_data):
        print(f"\n{Colors.BOLD}Sample {i+1}/{len(review_data)}:{Colors.ENDC}")
        print(f"{Colors.CYAN}Review: {data['text']}{Colors.ENDC}")
        print(f"Expected: {data['expected']}")
        
        try:
            # Format the prompt
            formatted_prompt = prompt_text.replace("{input_text}", data["text"])
            
            # Record prompt usage
            instance_id = optimizer.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=formatted_prompt
            )
            
            # Generate response
            print(f"{Colors.YELLOW}Generating classification...{Colors.ENDC}")
            response = llm_service.generate(formatted_prompt)
            response_text = response.lower().strip().replace('"', '') # Clean up the response
            print(f"{Colors.GREEN}Classification: {response_text}{Colors.ENDC}")
            
            # Record the response
            response_id = optimizer.record_response(
                prompt_instance_id=instance_id,
                content=response
            )
            
            # Determine if response is correct
            is_correct = response_text == data["expected"]
            
            # Record feedback
            feedback_id = optimizer.record_feedback(
                response_id=response_id,
                is_positive=is_correct,
                score=1.0 if is_correct else 0.0,
                comments=f"{'Correct' if is_correct else 'Incorrect'} classification"
            )
            
            print(f"{Colors.GREEN if is_correct else Colors.RED}✓ Recorded feedback: {'Correct' if is_correct else 'Incorrect'}{Colors.ENDC}")
            time.sleep(1)  # Pause between requests
        except Exception as e:
            print(f"{Colors.RED}Error processing sample {i+1}: {str(e)}{Colors.ENDC}")
    
    # Check optimization readiness and try to optimize
    try:
        result = check_and_optimize(optimizer, prompt_id, prompt_text, force=True)
        
        # If optimization was successful
        if result and isinstance(result, tuple) and len(result) == 2:
            new_prompt_id, optimized_text = result
            if new_prompt_id and optimized_text:
                # Test the optimized prompt
                print(f"\n{Colors.BOLD}Testing optimized prompt with new reviews:{Colors.ENDC}")
                
                # New test reviews
                test_reviews = [
                    "Room service was fast and the food was delicious.",
                    "Our room wasn't cleaned for two days despite requests."
                ]
                
                for review in test_reviews:
                    print(f"\n{Colors.CYAN}Review: {review}{Colors.ENDC}")
                    
                    try:
                        # Format with optimized prompt
                        formatted_optimized = optimized_text.replace("{input_text}", review)
                        print(f"{Colors.YELLOW}Generating classification...{Colors.ENDC}")
                        optimized_result = llm_service.generate(formatted_optimized)
                        print(f"{Colors.GREEN}Classification: {optimized_result.lower().strip()}{Colors.ENDC}")
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
        
        # Create a test prompt
        prompt_id, prompt_text = create_prompt(optimizer)
        
        # Choose mode
        print_section("Testing Mode")
        print("1. Interactive testing (provide feedback on each response)")
        print("2. Automatic demo (using sample reviews and expected classifications)")
        
        mode = input("\nSelect mode (1/2): ").strip()
        
        if mode == "2":
            automatic_demo(optimizer, llm_service, prompt_id, prompt_text)
        else:
            interactive_testing(optimizer, llm_service, prompt_id, prompt_text)
        
        print_header("Test Complete")
        print("Thank you for testing PromptCraft API!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\n{Colors.RED}Error during test: {str(e)}{Colors.ENDC}")
        raise

if __name__ == "__main__":
    main()