"""Streamlit interface for the Prompt Optimizer API."""

import streamlit as st
import re
import sys
import os
import logging
from datetime import datetime
import time
from dotenv import load_dotenv
from prompt_optimizer.utils.logging import setup_logging

load_dotenv()  # Load environment variables from .env file

# Add the project root to the Python path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
setup_logging(log_dir="./streamlit_logs")
logger = logging.getLogger(__name__)
# Import with proper error handling
try:
    from prompt_optimizer.api import PromptOptimizer
    from prompt_optimizer.config import OptimizerConfig
    from prompt_optimizer.response_objects import PromptResult, OptimizationResult, ValidationResult, OperationResult
    from prompt_optimizer.exceptions import (
        PromptCraftError, PromptNotFoundError, ResponseNotFoundError,
        OptimizationError, LLMError, ValidationError, StorageError
    )
except ImportError as e:
    st.error(f"❌ Failed to import PromptCraft components: {str(e)}")
    st.info("Please ensure all dependencies are installed and the project is properly set up.")
    st.stop()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state safely
def init_session_state():
    """Initialize session state variables."""
    try:
        defaults = {
            'feedback_history': [],
            'response_history': [],
            'monitored_prompts': {},
            'optimization_activity': [],
            'auto_optimizer_running': False,
            'error_count': 0
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
    except Exception as e:
        st.error(f"❌ Failed to initialize session state: {str(e)}")
        logger.error(f"Session state initialization failed: {str(e)}")

# Initialize session state
init_session_state()

# Safe optimizer initialization
@st.cache_resource
def get_optimizer(strategy_name="simple_ai"):
    """Get optimizer instance with proper error handling."""
    try:
        optimizer = PromptOptimizer(
            storage_dir="./streamlit_data",
            optimization_threshold=3,  # Lower threshold for demo purposes
            strategy_name=strategy_name
        )
        logger.info(f"Optimizer initialized successfully with strategy: {strategy_name}")
        return optimizer
        
    except StorageError as e:
        st.error(f"❌ Storage initialization failed: {str(e)}")
        st.info("💡 Try checking write permissions for ./streamlit_data directory")
        logger.error(f"Storage error during optimizer initialization: {str(e)}")
        st.stop()
        
    except ValidationError as e:
        st.error(f"❌ Configuration error: {str(e)}")
        logger.error(f"Validation error during optimizer initialization: {str(e)}")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Failed to initialize optimizer: {str(e)}")
        st.info("💡 This might be due to missing dependencies or configuration issues.")
        logger.error(f"Unexpected error during optimizer initialization: {str(e)}")
        st.stop()

# Get available strategies safely
def get_available_strategies():
    """Get available strategies with error handling."""
    try:
        return OptimizerConfig.available_strategies()
    except Exception as e:
        logger.warning(f"Failed to get available strategies: {str(e)}")
        return [{"name": "simple_ai", "description": "Simple AI strategy (fallback)"}]

# Error display helper
def show_error(message, error_type="error"):
    """Display errors consistently with proper styling."""
    if error_type == "error":
        st.error(f"❌ {message}")
    elif error_type == "warning":
        st.warning(f"⚠️ {message}")
    elif error_type == "info":
        st.info(f"ℹ️ {message}")

def show_success(message):
    """Display success messages consistently."""
    st.success(f"✅ {message}")

# Safe validation functions using new API
def safe_validate_prompt_id(prompt_id):
    """Validate prompt ID with comprehensive error handling."""
    try:
        if not prompt_id or not isinstance(prompt_id, str):
            return False, "Prompt ID must be a non-empty string"
        
        prompt_id = prompt_id.strip()
        if not prompt_id:
            return False, "Prompt ID cannot be empty or whitespace"
        
        result = optimizer.validate_prompt_id(prompt_id)
        return result.is_valid, result.message
        
    except Exception as e:
        logger.error(f"Unexpected error validating prompt ID {prompt_id}: {str(e)}")
        return False, f"Validation error: {str(e)}"

def safe_check_optimization_readiness(prompt_id):
    """Check optimization readiness with proper error handling."""
    try:
        if not prompt_id:
            return None, "Prompt ID is required"
        
        result = optimizer.get_optimization_stats(prompt_id)
        if result.success:
            return result.data, None
        else:
            return None, result.message
        
    except Exception as e:
        logger.error(f"Unexpected error checking optimization for {prompt_id}: {str(e)}")
        return None, f"Unexpected error: {str(e)}"

def safe_get_prompt(prompt_id):
    """Safely get prompt with error handling."""
    try:
        result = optimizer.get_prompt(prompt_id)
        if result.success:
            return result.data, None
        else:
            return None, result.message
    except Exception as e:
        logger.error(f"Error getting prompt {prompt_id}: {str(e)}")
        return None, f"Error retrieving prompt: {str(e)}"

# Initialize optimizer
try:
    optimizer = get_optimizer("simple_ai")
except Exception as e:
    st.error(f"❌ Critical initialization error: {str(e)}")
    st.stop()

# App title and description
st.title("PromptCraft API")
st.markdown("""
This interface allows you to test and visualize the prompt optimization process.
Create prompts, simulate responses, collect feedback, and see how the system automatically improves prompts.
""")

# Main navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Prompts", "Testing", "Feedback", "Optimization", "Interactive Testing", "Analytics", "Auto Optimization"
])

# Tab 1: Prompt Management
with tab1:
    st.header("Prompt Management")
    
    # Create a new prompt
    st.subheader("Create a New Prompt")
    new_prompt_text = st.text_area(
        "Prompt Template (use {placeholders} for variables)", 
        value="Summarize the following text in 2-3 sentences: {input_text}"
    )
    new_prompt_desc = st.text_input("Description", value="Text summarization prompt")
    
    if st.button("Create Prompt"):
        if not new_prompt_text.strip():
            show_error("Prompt text cannot be empty", "warning")
        elif len(new_prompt_text.strip()) < 5:
            show_error("Prompt text is too short (minimum 5 characters)", "warning")
        else:
            with st.spinner("Creating prompt..."):
                result = optimizer.register_prompt(
                    text=new_prompt_text.strip(), 
                    description=new_prompt_desc.strip()
                )
            
            if result.success:
                show_success(f"Prompt created with ID: {result.prompt_id}")
                st.session_state.last_created_prompt_id = result.prompt_id
                logger.info(f"Created prompt {result.prompt_id}")
            else:
                show_error(f"Failed to create prompt: {result.message}")
                if result.errors:
                    for error in result.errors:
                        st.error(f"  • {error}")
                st.session_state.error_count = st.session_state.get('error_count', 0) + 1
    
    # Display existing prompts
    st.subheader("Existing Prompts")
    
    try:
        with st.spinner("Loading prompts..."):
            all_prompts = optimizer.prompt_manager.list_prompts()
        
        if all_prompts:
            # Limit display for performance
            display_limit = 10
            displayed_prompts = all_prompts[:display_limit]
            
            if len(all_prompts) > display_limit:
                st.info(f"Showing {display_limit} of {len(all_prompts)} prompts")
            
            for i, p in enumerate(displayed_prompts):
                with st.expander(f"Prompt {i+1}: {p.id[:10]}...", expanded=i==0):
                    st.write(f"**ID:** {p.id}")
                    text_preview = p.text if len(p.text) < 100 else p.text[:100] + "..."
                    st.write(f"**Text:** {text_preview}")
                    st.write(f"**Version:** {p.version}")
                    st.write(f"**Description:** {p.description}")
                    
                    # Add copy ID button
                    if st.button(f"Copy ID", key=f"copy_{p.id}"):
                        st.code(p.id)
        else:
            st.info("No prompts found. Create your first prompt above.")
            
    except StorageError as e:
        show_error(f"Failed to load prompts: {str(e)}")
        st.info("💡 There might be an issue with the storage system.")
        
        # Show fallback with last created prompt
        if hasattr(st.session_state, "last_created_prompt_id"):
            st.info("Showing last created prompt as fallback:")
            prompt_data, error = safe_get_prompt(st.session_state.last_created_prompt_id)
            if prompt_data:
                st.code(f"ID: {st.session_state.last_created_prompt_id}\nText: {prompt_data['text']}\nDescription: {prompt_data['description']}")
            elif error:
                show_error(f"Could not load fallback prompt: {error}")
                
    except Exception as e:
        logger.error(f"Unexpected error loading prompts: {str(e)}")
        show_error(f"Failed to load prompts: {str(e)}")

# Tab 2: Testing Prompts
with tab2:
    st.header("Test Prompts")
    
    st.subheader("Select a Prompt")
    prompt_id_to_test = st.text_input(
        "Enter Prompt ID", 
        value=st.session_state.get("last_created_prompt_id", ""),
        key="test_prompt_id",
        help="Paste or type a prompt ID to test"
    )
    
    if prompt_id_to_test:
        is_valid, message = safe_validate_prompt_id(prompt_id_to_test)
        
        if is_valid:
            show_success(message)
            
            prompt_data, error = safe_get_prompt(prompt_id_to_test)
            
            if prompt_data:
                st.code(prompt_data['text'])
                
                # Extract placeholders safely
                try:
                    placeholders = re.findall(r'\{([^{}]+)\}', prompt_data['text'])
                except Exception as regex_error:
                    logger.warning(f"Regex error extracting placeholders: {str(regex_error)}")
                    placeholders = []
                    show_error("Could not extract placeholders from prompt", "warning")
                
                # Create input fields for each placeholder
                placeholder_values = {}
                if placeholders:
                    st.subheader("Fill in Placeholders")
                    for placeholder in placeholders:
                        placeholder_values[placeholder] = st.text_area(
                            f"Enter value for {{{placeholder}}}", 
                            value="This is a sample text that needs to be processed.",
                            key=f"placeholder_{placeholder}"
                        )
                
                # Format the prompt safely
                try:
                    formatted_prompt = prompt_data['text']
                    for placeholder, value in placeholder_values.items():
                        if value.strip():  # Only replace if value is not empty
                            formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", value)
                    
                    st.subheader("Formatted Prompt")
                    st.write(formatted_prompt)
                    
                except Exception as format_error:
                    logger.error(f"Error formatting prompt: {str(format_error)}")
                    show_error(f"Error formatting prompt: {str(format_error)}")
                    formatted_prompt = prompt_data['text']  # Fallback to original
                
                # Simulate AI response
                st.subheader("Simulate AI Response")
                ai_response = st.text_area(
                    "Enter or simulate the AI response", 
                    value="This is a simulated response from an AI system.",
                    height=100
                )
                
                if st.button("Record Usage and Response"):
                    if not ai_response.strip():
                        show_error("AI response cannot be empty", "warning")
                    else:
                        # Record prompt usage
                        with st.spinner("Recording prompt usage..."):
                            usage_result = optimizer.record_prompt_use(
                                prompt_id=prompt_id_to_test,
                                formatted_text=formatted_prompt
                            )
                        
                        if usage_result.success:
                            show_success(f"Recorded prompt usage (Instance ID: {usage_result.data['instance_id']})")
                            
                            # Record AI response
                            with st.spinner("Recording response..."):
                                response_result = optimizer.record_response(
                                    prompt_instance_id=usage_result.data['instance_id'],
                                    content=ai_response.strip()
                                )
                            
                            if response_result.success:
                                show_success(f"Recorded AI response (Response ID: {response_result.data['response_id']})")
                                
                                # Store for the feedback tab
                                st.session_state.last_response_id = response_result.data['response_id']
                                st.session_state.last_response_content = ai_response
                                st.session_state.current_prompt_id = prompt_id_to_test
                                
                                logger.info(f"Recorded usage for prompt {prompt_id_to_test}")
                            else:
                                show_error(f"Failed to record response: {response_result.message}")
                                if response_result.errors:
                                    for error in response_result.errors:
                                        st.error(f"  • {error}")
                        else:
                            show_error(f"Failed to record usage: {usage_result.message}")
                            if usage_result.errors:
                                for error in usage_result.errors:
                                    st.error(f"  • {error}")
                            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
                            
            elif error:
                show_error(error)
        else:
            show_error(message, "warning")

# Tab 3: Feedback Collection
with tab3:
    st.header("Provide Feedback")
    
    response_id_for_feedback = st.text_input(
        "Enter Response ID", 
        value=st.session_state.get("last_response_id", ""),
        help="Enter the response ID to provide feedback for"
    )
    
    if response_id_for_feedback:
        # Display the response content if available
        if (hasattr(st.session_state, "last_response_content") and 
            st.session_state.get("last_response_id") == response_id_for_feedback):
            st.subheader("Response Content")
            st.write(st.session_state.last_response_content)
            show_success("Valid response ID")
        
        # Collect feedback
        st.subheader("Rate this response")
        
        score = st.slider(
            "Feedback Score",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=1.0,
            help="Rate the quality of the response"
        )
        
        if st.button("Submit Feedback"):
            with st.spinner("Recording feedback..."):
                feedback_result = optimizer.record_feedback(
                    response_id=response_id_for_feedback,
                    score=score
                )
            
            if feedback_result.success:
                show_success(f"Feedback recorded with ID: {feedback_result.data['feedback_id']}")
                
                # Add to feedback history safely
                try:
                    if 'current_prompt_id' in st.session_state:
                        st.session_state.feedback_history.append({
                            "id": feedback_result.data['feedback_id'],
                            "prompt_id": st.session_state.current_prompt_id,
                            "response_id": response_id_for_feedback,
                            "score": score,
                            "timestamp": datetime.now()
                        })
                except Exception as history_error:
                    logger.warning(f"Failed to update feedback history: {str(history_error)}")
                
                # Check optimization readiness safely
                if 'current_prompt_id' in st.session_state:
                    prompt_id = st.session_state.current_prompt_id
                    readiness, error = safe_check_optimization_readiness(prompt_id)
                    
                    if readiness and readiness.get("is_ready"):
                        st.balloons()
                        show_success("🎉 This prompt is now ready for optimization!")
                    elif error:
                        show_error(f"Could not check optimization readiness: {error}", "warning")
                
                logger.info(f"Recorded feedback {feedback_result.data['feedback_id']} for response {response_id_for_feedback}")
            else:
                show_error(f"Failed to record feedback: {feedback_result.message}")
                if feedback_result.errors:
                    for error in feedback_result.errors:
                        st.error(f"  • {error}")
                st.session_state.error_count = st.session_state.get('error_count', 0) + 1

# Tab 4: Optimization
with tab4:
    st.header("Prompt Optimization")
    
    # Strategy selection
    st.subheader("Select Optimization Strategy")
    try:
        strategies = get_available_strategies()
        if strategies:
            strategy_options = {s["name"]: s["description"] for s in strategies}
            strategy_names = list(strategy_options.keys())
            selected_strategy = st.selectbox(
                "Optimization Strategy",
                options=strategy_names,
                format_func=lambda x: f"{x} - {strategy_options[x]}",
                help="Choose the optimization strategy to use"
            )
            
            # Update optimizer if strategy changed
            if 'current_strategy' not in st.session_state or st.session_state.current_strategy != selected_strategy:
                try:
                    st.session_state.current_strategy = selected_strategy
                    optimizer = get_optimizer(selected_strategy)
                    show_success(f"Using {selected_strategy} optimization strategy")
                except Exception as strategy_error:
                    logger.error(f"Failed to switch strategy: {str(strategy_error)}")
                    show_error(f"Failed to switch strategy: {str(strategy_error)}")
        else:
            st.info("Using default 'simple_ai' strategy")
            selected_strategy = "simple_ai"
            
    except Exception as e:
        logger.error(f"Error with strategy selection: {str(e)}")
        st.info("Using default 'simple_ai' strategy")
        selected_strategy = "simple_ai"
    
    # Select a prompt to optimize
    st.subheader("Select Prompt to Optimize")
    prompt_id_to_optimize = st.text_input(
        "Enter Prompt ID to optimize", 
        value=st.session_state.get("last_created_prompt_id", ""),
        help="Enter the ID of the prompt you want to optimize"
    )
    
    if prompt_id_to_optimize:
        is_valid, message = safe_validate_prompt_id(prompt_id_to_optimize)
        
        if is_valid:
            show_success(message)
            
            # Show the prompt
            prompt_data, error = safe_get_prompt(prompt_id_to_optimize)
            
            if prompt_data:
                st.subheader("Current Prompt")
                st.code(prompt_data['text'])
                
                # Get optimization readiness safely
                readiness, readiness_error = safe_check_optimization_readiness(prompt_id_to_optimize)
                
                if readiness:
                    st.subheader("Optimization Readiness")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Feedback Count", readiness.get("feedback_count", 0))
                    with col2:
                        st.metric("Threshold", readiness.get("threshold", 3))
                    with col3:
                        ready_status = "Yes" if readiness.get("is_ready", False) else "No"
                        st.metric("Ready for Optimization", ready_status)
                    
                    # Progress bar
                    try:
                        progress = min(readiness.get("feedback_count", 0) / max(readiness.get("threshold", 3), 1), 1.0)
                        st.progress(progress)
                    except Exception as progress_error:
                        logger.warning(f"Error creating progress bar: {str(progress_error)}")
                    
                    # Show strategy assessment if available
                    if "strategy_assessment" in readiness:
                        with st.expander("Strategy Assessment Details"):
                            st.json(readiness["strategy_assessment"])
                    
                    if readiness.get("is_ready", False):
                        show_success("This prompt is ready for optimization!")
                    else:
                        needed = readiness.get('threshold', 3) - readiness.get('feedback_count', 0)
                        st.warning(f"Need {needed} more feedback items")
                    
                    # Optimization controls
                    st.subheader("Optimization Controls")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        force_optimize = st.checkbox(
                            "Force optimization (even if not ready)",
                            help="Override readiness check and optimize anyway"
                        )
                    
                    with col2:
                        if st.button("Optimize Prompt", type="primary"):
                            with st.spinner("Generating optimization..."):
                                opt_result = optimizer.optimize_prompt(
                                    prompt_id=prompt_id_to_optimize,
                                    force=force_optimize
                                )
                            
                            if opt_result.success:
                                show_success("Optimization successful!")
                                
                                if opt_result.optimization_applied:
                                    st.subheader("New Optimized Prompt Created")
                                    st.info(f"New prompt ID: {opt_result.new_prompt_id}")
                                    
                                    # Try to show the new prompt
                                    new_prompt_data, new_error = safe_get_prompt(opt_result.new_prompt_id)
                                    if new_prompt_data:
                                        st.subheader("Optimized Prompt")
                                        st.code(new_prompt_data['text'])
                                        
                                        # Show comparison
                                        st.subheader("Comparison")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Original:**")
                                            st.code(prompt_data['text'])
                                        with col2:
                                            st.write("**Optimized:**")
                                            st.code(new_prompt_data['text'])
                                        
                                        # Store the new prompt ID
                                        st.session_state.last_created_prompt_id = opt_result.new_prompt_id
                                    else:
                                        st.info("Optimization created but could not display details.")
                                else:
                                    # Generated optimization but not applied
                                    st.subheader("Generated Optimization")
                                    if opt_result.data and 'optimized_text' in opt_result.data:
                                        st.code(opt_result.data['optimized_text'])
                                        
                                        # Show comparison
                                        st.subheader("Comparison")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Original:**")
                                            st.code(prompt_data['text'])
                                        with col2:
                                            st.write("**Optimized (not applied):**")
                                            st.code(opt_result.data['optimized_text'])
                                    else:
                                        st.info("Optimization generated but text not available.")
                            else:
                                show_error(f"Optimization failed: {opt_result.message}")
                                if opt_result.errors:
                                    for error in opt_result.errors:
                                        st.error(f"  • {error}")
                                
                                # Show readiness info if available
                                if opt_result.readiness_info:
                                    st.info("Readiness information:")
                                    st.json(opt_result.readiness_info)
                    
                    # Version history
                    st.subheader("Prompt Version History")
                    try:
                        with st.spinner("Loading version history..."):
                            history = optimizer.optimizer.get_optimization_history(prompt_id_to_optimize)
                        
                        if history:
                            for i, version in enumerate(history):
                                with st.expander(f"Version {version.get('version', '?')}", expanded=i==0):
                                    st.write(f"**ID:** {version.get('prompt_id', 'Unknown')}")
                                    st.write(f"**Created:** {version.get('created_at', 'Unknown')}")
                                    st.write(f"**Text:** {version.get('text', 'Unknown')[:100]}...")
                                    if version.get('description'):
                                        st.write(f"**Description:** {version.get('description')}")
                        else:
                            st.info("No optimization history yet")
                            
                    except Exception as history_error:
                        logger.warning(f"Could not load version history: {str(history_error)}")
                        st.info("Version history not available")
                        
                elif readiness_error:
                    show_error(f"Cannot check optimization readiness: {readiness_error}")
                    
            elif error:
                show_error(error)
        else:
            show_error(message, "warning")

# Simplified versions of remaining tabs
# Tab 5: Interactive Testing
with tab5:
    st.header("Interactive Testing Environment")
    st.info("Interactive testing with comprehensive error handling")
    
    # Basic interactive testing functionality
    st.subheader("Quick Test")
    
    if st.session_state.get("last_created_prompt_id"):
        prompt_id = st.session_state.last_created_prompt_id
        
        # Get prompt for testing
        prompt_data, error = safe_get_prompt(prompt_id)
        if prompt_data:
            st.write(f"**Current Prompt:** {prompt_data['text']}")
            
            # Quick test input
            test_input = st.text_input("Enter test input:")
            
            if test_input and st.button("Quick Test"):
                formatted_text = prompt_data['text'].replace("{input_text}", test_input)
                
                # Record usage
                usage_result = optimizer.record_prompt_use(prompt_id, formatted_text)
                if usage_result.success:
                    st.success("Test recorded successfully!")
                    st.code(formatted_text)
                else:
                    show_error(f"Failed to record test: {usage_result.message}")
        else:
            st.info("Create a prompt first to use interactive testing")
    else:
        st.info("Create a prompt first to use interactive testing")

# Tab 6: Analytics
with tab6:
    st.header("Analytics & Visualization")
    st.info("Analytics with proper error handling")
    
    # Show feedback history
    if st.session_state.feedback_history:
        st.subheader("Feedback History")
        
        for i, feedback in enumerate(st.session_state.feedback_history[-5:]):  # Show last 5
            with st.expander(f"Feedback {i+1} - Score {feedback['score']}"):
                st.write(f"**Prompt ID:** {feedback['prompt_id']}")
                st.write(f"**Time:** {feedback['timestamp']}")
    else:
        st.info("No feedback data available yet")

# Tab 7: Auto Optimization
with tab7:
    st.header("Auto Optimization")
    st.info("Auto optimization with comprehensive error handling")
    
    # Basic auto optimization controls
    st.subheader("Auto Optimization Status")
    
    if st.button("Check All Prompts for Optimization"):
        try:
            all_prompts = optimizer.prompt_manager.list_prompts()
            
            if all_prompts:
                st.write("**Optimization Status:**")
                
                for prompt in all_prompts[:5]:  # Check first 5 prompts
                    readiness, error = safe_check_optimization_readiness(prompt.id)
                    
                    if readiness:
                        status = "✅ Ready" if readiness.get("is_ready") else "⏳ Not Ready"
                        feedback_count = readiness.get("feedback_count", 0)
                        threshold = readiness.get("threshold", 3)
                        
                        st.write(f"**{prompt.id[:12]}...** - {status} ({feedback_count}/{threshold} feedback)")
                    else:
                        st.write(f"**{prompt.id[:12]}...** - ❌ Error: {error}")
            else:
                st.info("No prompts available for optimization")
                
        except Exception as e:
            show_error(f"Failed to check optimization status: {str(e)}")

# Footer and error tracking
st.markdown("---")
st.caption("PromptCraft API Demo Interface")

# Error rate tracking
error_count = st.session_state.get('error_count', 0)
if error_count > 3:
    st.sidebar.warning(f"⚠️ {error_count} errors detected during this session")
    if st.sidebar.button("Reset Error Count"):
        st.session_state.error_count = 0
        st.rerun()

# Debug info (only show if there have been errors)
if error_count > 0:
    with st.sidebar.expander("Debug Information"):
        st.write(f"Errors this session: {error_count}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")
        
        if st.button("Clear All Session Data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()