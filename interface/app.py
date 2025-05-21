"""Streamlit interface for the Prompt Optimizer API."""

import streamlit as st
import re
import sys
import os
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Add the project root to the Python path so we can import the package
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Prompt Optimizer API
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.config import OptimizerConfig

# Initialize feedback history in session state if not present
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Initialize response history for interactive testing
if 'response_history' not in st.session_state:
    st.response_history = []

# Initialize the Prompt Optimizer (we'll use a demo directory)
@st.cache_resource
def get_optimizer(strategy_name="simple_ai"):
    return PromptOptimizer(
        storage_dir="./streamlit_data",
        optimization_threshold=3,  # Lower threshold for demo purposes
        strategy_name=strategy_name
    )

# Default to SimpleAI strategy as it doesn't require a reward model
optimizer = get_optimizer("simple_ai")

# Get available strategies from config
def get_available_strategies():
    return OptimizerConfig.available_strategies()

# App title and description
st.title("PromptCraft API")
st.markdown("""
This interface allows you to test and visualize the prompt optimization process.
Create prompts, simulate responses, collect feedback, and see how the system automatically improves prompts.
""")

# Helper function to validate prompt IDs
def validate_prompt_id(prompt_id):
    if not prompt_id:
        return False
    return optimizer.validate_prompt_id(prompt_id)

# Helper function to check if prompt needs optimization
def check_optimization_readiness(prompt_id):
    # Use actual optimizer engine readiness check instead of simple count
    try:
        return optimizer.get_optimization_stats(prompt_id)
    except:
        # Fallback to simple check if optimizer method fails
        feedback_count = len([f for f in st.session_state.feedback_history 
                            if f.get('prompt_id') == prompt_id])
        threshold = 3  # Hardcoded for demo
        is_ready = feedback_count >= threshold
        return {
            "prompt_id": prompt_id,
            "feedback_count": feedback_count,
            "threshold": threshold,
            "is_ready": is_ready
        }

# Main navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Prompts", "Testing", "Feedback", "Optimization", "Interactive Testing", "Analytics", "Auto Optimization"
])

# Tab 1: Prompt Management
with tab1:
    st.header("Prompt Management")
    
    # Create a new prompt
    st.subheader("Create a New Prompt")
    new_prompt_text = st.text_area("Prompt Template (use {placeholders} for variables)", 
                                  value="Summarize the following text in 2-3 sentences: {input_text}")
    new_prompt_desc = st.text_input("Description", value="Text summarization prompt")
    
    if st.button("Create Prompt"):
        prompt_id = optimizer.register_prompt(text=new_prompt_text, description=new_prompt_desc)
        st.success(f"Prompt created with ID: {prompt_id}")
        st.session_state.last_created_prompt_id = prompt_id
    
    # Display existing prompts
    st.subheader("Existing Prompts")
    
    # Attempt to list all prompts if possible
    try:
        all_prompts = optimizer.prompt_manager.list_prompts()
        if all_prompts:
            # Display a list of prompts without using dataframes
            for i, p in enumerate(all_prompts):
                with st.expander(f"Prompt {i+1}: {p.id[:10]}...", expanded=i==0):
                    st.write(f"**ID:** {p.id}")
                    st.write(f"**Text:** {p.text if len(p.text) < 100 else p.text[:100] + '...'}")
                    st.write(f"**Version:** {p.version}")
                    st.write(f"**Description:** {p.description}")
    except:
        # Fallback to showing just the last created prompt
        if hasattr(st.session_state, "last_created_prompt_id"):
            prompt_id = st.session_state.last_created_prompt_id
            prompt = optimizer.get_prompt(prompt_id)
            if prompt:
                st.code(f"ID: {prompt_id}\nText: {prompt['text']}\nDescription: {prompt['description']}")

# Tab 2: Testing Prompts
with tab2:
    st.header("Test Prompts")
    
    # Select a prompt to test
    st.subheader("Select a Prompt")
    prompt_id_to_test = st.text_input("Enter Prompt ID", 
                                     value=st.session_state.get("last_created_prompt_id", ""),
                                     key="test_prompt_id")
    
    # Validate the prompt ID
    if prompt_id_to_test:
        if validate_prompt_id(prompt_id_to_test):
            st.success("✓ Valid prompt ID")
            prompt = optimizer.get_prompt(prompt_id_to_test)
            st.code(prompt['text'])
            
            # Extract placeholders from the prompt
            placeholders = re.findall(r'\{([^{}]+)\}', prompt['text'])
            
            # Create input fields for each placeholder
            placeholder_values = {}
            for placeholder in placeholders:
                placeholder_values[placeholder] = st.text_area(f"Enter value for {{{placeholder}}}", 
                                                             value="This is a sample text that needs to be processed.")
            
            # Format the prompt with the provided values
            formatted_prompt = prompt['text']
            for placeholder, value in placeholder_values.items():
                formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", value)
            
            st.subheader("Formatted Prompt")
            st.write(formatted_prompt)
            
            # Simulate AI response
            st.subheader("Simulate AI Response")
            ai_response = st.text_area("Enter or simulate the AI response", 
                                     value="This is a simulated response from an AI system.")
            
            if st.button("Record Usage and Response"):
                # Record prompt usage
                with st.spinner("Recording prompt usage..."):
                    instance_id = optimizer.record_prompt_use(
                        prompt_id=prompt_id_to_test,
                        formatted_text=formatted_prompt
                    )
                
                # Record AI response
                with st.spinner("Recording response..."):
                    response_id = optimizer.record_response(
                        prompt_instance_id=instance_id,
                        content=ai_response
                    )
                
                st.success(f"Recorded prompt usage (Instance ID: {instance_id})")
                st.success(f"Recorded AI response (Response ID: {response_id})")
                
                # Store for the feedback tab
                st.session_state.last_response_id = response_id
                st.session_state.last_response_content = ai_response
                st.session_state.current_prompt_id = prompt_id_to_test
        else:
            st.error("❌ Invalid prompt ID. Please enter a valid ID.")

# Tab 3: Feedback Collection
with tab3:
    st.header("Provide Feedback")
    
    # Select or enter a response ID
    response_id_for_feedback = st.text_input("Enter Response ID", 
                                           value=st.session_state.get("last_response_id", ""))
    
    if response_id_for_feedback:
        try:
            # Try to validate the response ID
            # Display the response content if available
            if hasattr(st.session_state, "last_response_content") and st.session_state.last_response_id == response_id_for_feedback:
                st.subheader("Response Content")
                st.write(st.session_state.last_response_content)
                st.success("✓ Valid response ID")
            
            # Collect feedback
            st.subheader("Rate this response")
            is_positive = st.radio("Was this response good?", options=["Yes", "No"]) == "Yes"
            score = st.slider("Score (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            comments = st.text_area("Comments", value="")
            
            if st.button("Submit Feedback"):
                with st.spinner("Recording feedback..."):
                    try:
                        feedback_id = optimizer.record_feedback(
                            response_id=response_id_for_feedback,
                            is_positive=is_positive,
                            score=score,
                            comments=comments
                        )
                        st.success(f"Feedback recorded with ID: {feedback_id}")
                        
                        # Add to feedback history in session state 
                        # This is a simple way to track feedback without a database
                        if 'current_prompt_id' in st.session_state:
                            st.session_state.feedback_history.append({
                                "id": feedback_id,
                                "prompt_id": st.session_state.current_prompt_id,
                                "response_id": response_id_for_feedback,
                                "is_positive": is_positive,
                                "score": score,
                                "comments": comments,
                                "timestamp": datetime.now()
                            })
                        
                        # Show refreshed stats after short delay to let the system update
                        time.sleep(0.5)  # Give a moment for stats to update
                        
                        # Show the current feedback count
                        if 'current_prompt_id' in st.session_state:
                            prompt_id = st.session_state.current_prompt_id
                            st.info(f"This prompt now has {len(st.session_state.feedback_history)} feedback items.")
                            
                            # Check optimization readiness
                            readiness = check_optimization_readiness(prompt_id)
                            if readiness["is_ready"]:
                                st.success(f"🎉 This prompt is now ready for optimization!")
                        
                    except ValueError as e:
                        st.error(f"Error recording feedback: {str(e)}")
        except:
            st.error("❌ Invalid response ID. Please enter a valid ID.")

# Tab 4: Optimization
with tab4:
    st.header("Prompt Optimization")
    
    # Strategy selection
    st.subheader("Select Optimization Strategy")
    try:
        strategies = get_available_strategies()
        strategy_options = {s["name"]: s["description"] for s in strategies}
        strategy_names = list(strategy_options.keys())
        selected_strategy = st.selectbox(
            "Optimization Strategy",
            options=strategy_names,
            format_func=lambda x: f"{x} - {strategy_options[x]}"
        )
        
        # Update optimizer if strategy changed
        if 'current_strategy' not in st.session_state or st.session_state.current_strategy != selected_strategy:
            st.session_state.current_strategy = selected_strategy
            optimizer = get_optimizer(selected_strategy)
            st.success(f"Using {selected_strategy} optimization strategy")
    except:
        st.info("Using default 'simple_ai' strategy")
        selected_strategy = "simple_ai"
    
    # Select a prompt to optimize
    prompt_id_to_optimize = st.text_input("Enter Prompt ID to optimize", 
                                        value=st.session_state.get("last_created_prompt_id", ""))
    
    if prompt_id_to_optimize:
        if validate_prompt_id(prompt_id_to_optimize):
            st.success("✓ Valid prompt ID")
            
            # Show the prompt
            prompt = optimizer.get_prompt(prompt_id_to_optimize)
            st.subheader("Current Prompt")
            st.code(prompt['text'])
            
            # Get optimization readiness
            readiness = check_optimization_readiness(prompt_id_to_optimize)
            
            # Display optimization stats
            st.subheader("Optimization Readiness")
            
            # Create metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Feedback Count", readiness.get("feedback_count", 0))
            with col2:
                st.metric("Threshold", readiness.get("threshold", 3))
            with col3:
                st.metric("Ready for Optimization", "Yes" if readiness.get("is_ready", False) else "No")
            
            # Display progress bar
            progress = min(readiness.get("feedback_count", 0) / readiness.get("threshold", 3), 1.0)
            st.progress(progress)
            
            # Show strategy assessment if available
            if "strategy_assessment" in readiness:
                st.subheader("Strategy Assessment")
                st.json(readiness["strategy_assessment"])
            
            if readiness.get("is_ready", False):
                st.success("✓ This prompt is ready for optimization!")
            else:
                st.warning(f"Need {readiness.get('threshold', 3) - readiness.get('feedback_count', 0)} more feedback items")
            
            # Optimization button
            force_optimize = st.checkbox("Force optimization (even if not ready)")
            
            if st.button("Optimize Prompt"):
                with st.spinner("Generating optimization..."):
                    try:
                        # Use the actual optimizer engine
                        result = optimizer.optimize_prompt(
                            prompt_id=prompt_id_to_optimize,
                            force=force_optimize
                        )
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        result = None
                    
                    # Fallback to simple optimization if the actual one fails
                    if result is None and force_optimize:
                        st.warning("Falling back to placeholder optimization")
                        current_text = prompt['text']
                        optimized_text = "Be specific and " + current_text
                        
                        # Update the prompt to create a new version
                        result = optimizer.prompt_manager.update_prompt(
                            prompt_id=prompt_id_to_optimize,
                            text=optimized_text
                        ).id
                
                if result:
                    st.success(f"Optimization applied! New prompt ID: {result}")
                    new_prompt = optimizer.get_prompt(result)
                    
                    # Display comparison
                    st.subheader("Original Prompt")
                    original_prompt = optimizer.get_prompt(prompt_id_to_optimize)
                    st.code(original_prompt['text'])
                    
                    st.subheader("Optimized Prompt")
                    st.code(new_prompt['text'])
                    
                    # Store the new prompt ID
                    st.session_state.last_created_prompt_id = result
                    
                    # Display diff if possible
                    try:
                        import difflib
                        d = difflib.Differ()
                        diff = list(d.compare(original_prompt['text'].splitlines(), new_prompt['text'].splitlines()))
                        st.subheader("Prompt Differences")
                        st.code("\n".join(diff))
                    except:
                        pass
                else:
                    st.error("Could not generate optimization. Make sure you have enough feedback or try forcing optimization.")
            
            # Version history
            st.subheader("Prompt Version History")
            try:
                history = optimizer.optimizer.get_optimization_history(prompt_id_to_optimize)
                if history:
                    # Display history without dataframes
                    for i, version in enumerate(history):
                        st.write(f"**Version {version.get('version', '?')}** - {version.get('created_at', '').strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"ID: {version.get('prompt_id', 'Unknown')}")
                        st.write(f"Text: {version.get('text', 'Unknown')[:100]}...")
                        st.write("---")
                else:
                    st.info("No optimization history yet")
            except Exception as e:
                st.info("Version history not available")
                
        else:
            st.error("❌ Invalid prompt ID. Please enter a valid ID.")

# Tab 5: Interactive Testing - New tab that combines all functionality in a single workflow
with tab5:
    st.header("Interactive Testing Environment")
    
    # Select or display the current prompt
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Prompt")
        
        if not hasattr(st.session_state, "active_prompt_id"):
            st.session_state.active_prompt_id = st.session_state.get("last_created_prompt_id", "")
            
        prompt_id_input = st.text_input("Enter Prompt ID to test", 
                                     value=st.session_state.active_prompt_id,
                                     key="interactive_prompt_id")
        
        if prompt_id_input and prompt_id_input != st.session_state.active_prompt_id:
            if validate_prompt_id(prompt_id_input):
                st.session_state.active_prompt_id = prompt_id_input
                st.session_state.response_history = []  # Reset history when prompt changes
                st.success("✓ Prompt loaded successfully")
                st.rerun()
            else:
                st.error("❌ Invalid prompt ID")
        
        # Display the active prompt
        if hasattr(st.session_state, "active_prompt_id") and st.session_state.active_prompt_id:
            prompt = optimizer.get_prompt(st.session_state.active_prompt_id)
            if prompt:
                with st.expander("Prompt Template", expanded=True):
                    st.code(prompt['text'])
                
                # Extract placeholders
                placeholders = re.findall(r'\{([^{}]+)\}', prompt['text'])
                
                # Create input form
                st.subheader("Test Input")
                with st.form(key="prompt_test_form"):
                    # Input fields for each placeholder
                    placeholder_values = {}
                    for placeholder in placeholders:
                        placeholder_values[placeholder] = st.text_area(
                            f"Enter {placeholder}", 
                            value="This is a sample text for testing the optimization process."
                        )
                    
                    # Format the prompt with the provided values
                    formatted_prompt = prompt['text']
                    for placeholder, value in placeholder_values.items():
                        formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", value)
                    
                    # Submit button
                    submit_button = st.form_submit_button(label="Generate Response")
                    
                if submit_button:
                    # Record prompt usage and simulate response
                    with st.spinner("Generating response..."):
                        # Record prompt instance
                        instance_id = optimizer.record_prompt_use(
                            prompt_id=st.session_state.active_prompt_id,
                            formatted_text=formatted_prompt
                        )
                        
                        # Simulate AI response (in a real app, this would call an AI service)
                        simulated_response = f"This is a simulated response for: '{formatted_prompt[:30]}...' (Generated at {datetime.now().strftime('%H:%M:%S')})"
                        
                        # Record response
                        response_id = optimizer.record_response(
                            prompt_instance_id=instance_id,
                            content=simulated_response
                        )
                        
                        # Add to response history
                        if not hasattr(st.session_state, "response_history"):
                            st.session_state.response_history = []
                            
                        st.session_state.response_history.append({
                            "prompt_id": st.session_state.active_prompt_id,
                            "instance_id": instance_id,
                            "response_id": response_id,
                            "input": formatted_prompt[:50] + "..." if len(formatted_prompt) > 50 else formatted_prompt,
                            "response": simulated_response,
                            "timestamp": datetime.now(),
                            "feedback_given": False
                        })
                        
                    st.success("Response generated!")
                    st.rerun()
    
    with col2:
        st.subheader("Optimization Status")
        
        if hasattr(st.session_state, "active_prompt_id") and st.session_state.active_prompt_id:
            # Check if prompt is ready for optimization
            stats = check_optimization_readiness(st.session_state.active_prompt_id)
            
            # Display optimization readiness
            st.write(f"Feedback count: {stats.get('feedback_count', 0)}/{stats.get('threshold', 3)}")
            
            # Progress bar
            progress = min(stats.get('feedback_count', 0) / stats.get('threshold', 3), 1.0)
            st.progress(progress)
            
            if stats.get('is_ready', False):
                st.success("✓ Ready for optimization!")
                
                # Strategy selection
                try:
                    strategies = get_available_strategies()
                    strategy_names = [s["name"] for s in strategies]
                    selected_strategy = st.selectbox(
                        "Optimization Strategy",
                        options=strategy_names,
                        key="interactive_strategy"
                    )
                except:
                    selected_strategy = "simple_ai"
                
                # Add auto-optimize option
                if st.button("Apply Optimization"):
                    with st.spinner("Optimizing prompt..."):
                        try:
                            # Update optimizer with selected strategy
                            optimizer = get_optimizer(selected_strategy)
                            
                            # Use the actual optimizer engine
                            result = optimizer.optimize_prompt(
                                prompt_id=st.session_state.active_prompt_id,
                                force=True  # Force optimization
                            )
                            
                            # Use result if available
                            if result:
                                new_prompt_id = result
                            else:
                                # Fallback to placeholder
                                prompt = optimizer.get_prompt(st.session_state.active_prompt_id)
                                current_text = prompt['text']
                                optimized_text = "Be specific and " + current_text
                                
                                # Create new version
                                new_prompt_id = optimizer.prompt_manager.update_prompt(
                                    prompt_id=st.session_state.active_prompt_id,
                                    text=optimized_text
                                ).id
                            
                            # Update the active prompt
                            st.session_state.active_prompt_id = new_prompt_id
                            st.session_state.response_history = []  # Reset responses
                            
                        except Exception as e:
                            st.error(f"Error during optimization: {str(e)}")
                    
                    st.success("✓ Prompt optimized!")
                    st.rerun()
            else:
                st.info(f"Need {stats.get('threshold', 3) - stats.get('feedback_count', 0)} more feedback items")
    
    # Response history and feedback
    if hasattr(st.session_state, "response_history") and st.session_state.response_history:
        st.header("Response History")
        
        for i, response_item in enumerate(reversed(st.session_state.response_history)):
            with st.expander(f"Response {len(st.session_state.response_history) - i}", expanded=i == 0):
                st.write("**Input:**", response_item["input"])
                st.write("**Response:**", response_item["response"])
                st.write("**Time:**", response_item["timestamp"].strftime("%H:%M:%S"))
                
                # Feedback section
                if not response_item.get("feedback_given", False):
                    st.divider()
                    st.subheader("Provide Feedback")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        is_positive = st.radio(
                            "Was this response good?", 
                            options=["Yes", "No"], 
                            key=f"rating_{response_item['response_id']}"
                        ) == "Yes"
                    
                    with col2:
                        score = st.slider(
                            "Score (0-1)", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.7, 
                            step=0.1,
                            key=f"score_{response_item['response_id']}"
                        )
                    
                    comments = st.text_area(
                        "Comments", 
                        value="",
                        key=f"comments_{response_item['response_id']}"
                    )
                    
                    if st.button("Submit Feedback", key=f"submit_{response_item['response_id']}"):
                        with st.spinner("Recording feedback..."):
                            # Record feedback
                            feedback_id = optimizer.record_feedback(
                                response_id=response_item['response_id'],
                                is_positive=is_positive,
                                score=score,
                                comments=comments
                            )
                            
                            # Add to feedback history
                            st.session_state.feedback_history.append({
                                "id": feedback_id,
                                "prompt_id": response_item['prompt_id'],
                                "response_id": response_item['response_id'],
                                "is_positive": is_positive,
                                "score": score,
                                "comments": comments,
                                "timestamp": datetime.now()
                            })
                            
                            # Mark as feedback given
                            for item in st.session_state.response_history:
                                if item['response_id'] == response_item['response_id']:
                                    item['feedback_given'] = True
                            
                            # Check if optimization is now possible
                            stats = check_optimization_readiness(response_item['prompt_id'])
                            if stats.get('is_ready', False):
                                st.success("🎉 This prompt is now ready for optimization!")
                        
                        st.success("Thank you for your feedback!")
                        st.rerun()
                else:
                    st.info("✓ Feedback provided")
                                
    else:
        st.info("No responses yet. Test the prompt to generate responses.")

# Tab 6: Analytics and Visualization (Simplified)
with tab6:
    st.header("Analytics & Visualization")
    
    # Prompt selection
    prompt_id_for_analytics = st.text_input("Enter Prompt ID for analytics", 
                                         value=st.session_state.get("last_created_prompt_id", ""))
    
    if prompt_id_for_analytics and validate_prompt_id(prompt_id_for_analytics):
        st.success("✓ Valid prompt ID")
        
        st.subheader("Prompt Information")
        prompt = optimizer.get_prompt(prompt_id_for_analytics)
        if prompt:
            st.json(prompt)
            
            # Try to get parent prompt if it exists
            if prompt.get("parent_id"):
                st.subheader("Parent Prompt")
                parent = optimizer.get_prompt(prompt.get("parent_id"))
                if parent:
                    st.json(parent)
                    
                    # Simple text comparison
                    st.subheader("Text Comparison")
                    st.text("Original Prompt:")
                    st.code(parent.get("text", ""))
                    st.text("Current Prompt:")
                    st.code(prompt.get("text", ""))
            
            # Simple feedback summary
            st.subheader("Feedback Summary")
            try:
                # Get feedback data from session state (won't rely on potentially problematic libraries)
                feedback_items = [f for f in st.session_state.feedback_history 
                                if f.get('prompt_id') == prompt_id_for_analytics]
                
                if feedback_items:
                    positive_count = sum(1 for f in feedback_items if f.get('is_positive', False))
                    total_count = len(feedback_items)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Feedback", total_count)
                    with col2:
                        st.metric("Positive Feedback", positive_count)
                    with col3:
                        positive_rate = positive_count / total_count if total_count > 0 else 0
                        st.metric("Positive Rate", f"{positive_rate:.1%}")
                    
                    # Display recent feedback
                    st.subheader("Recent Feedback")
                    recent_feedback = sorted(feedback_items, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)[:5]
                    
                    for i, feedback in enumerate(recent_feedback):
                        with st.expander(f"Feedback {i+1}", expanded=i==0):
                            st.write(f"**Score:** {feedback.get('score', 'N/A')}")
                            st.write(f"**Positive:** {'Yes' if feedback.get('is_positive', False) else 'No'}")
                            if feedback.get('comments'):
                                st.write(f"**Comments:** {feedback.get('comments')}")
                            st.write(f"**Time:** {feedback.get('timestamp', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.info("No feedback data available for this prompt")
                
            except Exception as e:
                st.error(f"Error processing feedback data: {str(e)}")
    else:
        st.info("Enter a valid prompt ID to view analytics")

# Tab 7: Auto Optimization
with tab7:
    st.header("Auto Optimization")
    st.markdown("""
    This tab allows you to test the auto-optimization feature, which continuously monitors 
    prompts and applies optimizations when enough feedback is gathered.
    """)
    
    # Import auto optimizer
    try:
        from prompt_optimizer.auto_optimizer import AutoOptimizer
    except ImportError:
        st.error("AutoOptimizer module couldn't be imported. Make sure all dependencies are installed.")
    
    # Initialize auto-optimizer in session state if not present
    if 'auto_optimizer' not in st.session_state:
        try:
            st.session_state.auto_optimizer = AutoOptimizer()
            st.session_state.monitored_prompts = {}
            st.session_state.optimization_activity = []
            st.session_state.auto_optimizer_running = False
        except Exception as e:
            st.error(f"Error initializing AutoOptimizer: {str(e)}")
    
    # Configuration section
    st.subheader("Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        check_interval = st.number_input("Check Interval (seconds)", 
                                        min_value=10, 
                                        max_value=3600, 
                                        value=30)
    
    with config_col2:
        try:
            strategies = OptimizerConfig.available_strategies()
            strategy_names = [s["name"] for s in strategies]
            auto_strategy = st.selectbox(
                "Optimization Strategy",
                options=strategy_names,
                index=strategy_names.index("simple_ai") if "simple_ai" in strategy_names else 0,
                key="auto_strategy"
            )
        except:
            auto_strategy = "simple_ai"
            st.info("Using default 'simple_ai' strategy")
    
    # Prompt selection
    st.subheader("Monitored Prompts")
    
    # Add prompts to monitor
    prompt_to_monitor = st.text_input("Enter Prompt ID to monitor")
    
    if prompt_to_monitor:
        if validate_prompt_id(prompt_to_monitor):
            if st.button("Add to Monitoring"):
                try:
                    # Add to session state
                    if prompt_to_monitor not in st.session_state.monitored_prompts:
                        prompt = optimizer.get_prompt(prompt_to_monitor)
                        st.session_state.monitored_prompts[prompt_to_monitor] = {
                            "id": prompt_to_monitor,
                            "text": prompt["text"][:50] + "..." if len(prompt["text"]) > 50 else prompt["text"],
                            "last_check": None,
                            "last_optimized": None
                        }
                        
                        # Add to auto optimizer
                        if hasattr(st.session_state, "auto_optimizer"):
                            st.session_state.auto_optimizer.monitored_prompts[prompt_to_monitor] = datetime.now()
                        
                        st.success(f"Added prompt {prompt_to_monitor} to monitoring")
                    else:
                        st.warning("This prompt is already being monitored")
                except Exception as e:
                    st.error(f"Error adding prompt to monitoring: {str(e)}")
        else:
            st.error("Invalid prompt ID")
    
    # Display monitored prompts without dataframes
    if st.session_state.monitored_prompts:
        # Display each monitored prompt in an expander
        for prompt_id, data in st.session_state.monitored_prompts.items():
            with st.expander(f"Prompt: {prompt_id[:8]}...", expanded=True):
                st.write(f"**ID:** {prompt_id}")
                st.write(f"**Text:** {data['text']}")
                st.write(f"**Last Check:** {data['last_check'].strftime('%H:%M:%S') if data['last_check'] else 'Never'}")
                st.write(f"**Last Optimized:** {data['last_optimized'].strftime('%H:%M:%S') if data['last_optimized'] else 'Never'}")
        
        # Option to remove prompts
        prompt_to_remove = st.selectbox(
            "Select prompt to remove from monitoring",
            options=list(st.session_state.monitored_prompts.keys())
        )
        
        if st.button("Remove from Monitoring"):
            try:
                if prompt_to_remove in st.session_state.monitored_prompts:
                    del st.session_state.monitored_prompts[prompt_to_remove]
                    
                    # Remove from auto optimizer
                    if hasattr(st.session_state, "auto_optimizer") and prompt_to_remove in st.session_state.auto_optimizer.monitored_prompts:
                        del st.session_state.auto_optimizer.monitored_prompts[prompt_to_remove]
                    
                    st.success(f"Removed prompt {prompt_to_remove} from monitoring")
                    st.rerun()
            except Exception as e:
                st.error(f"Error removing prompt: {str(e)}")
    else:
        st.info("No prompts are currently being monitored. Add a prompt above.")
    
    # Auto-optimization controls
    st.subheader("Controls")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        if not st.session_state.auto_optimizer_running:
            start_button = st.button("Start Auto-Optimization")
            if start_button and st.session_state.monitored_prompts:
                try:
                    # Configure auto optimizer
                    st.session_state.auto_optimizer.check_interval = check_interval
                    st.session_state.auto_optimizer.config.strategy_name = auto_strategy
                    
                    # Start the background thread
                    st.session_state.auto_optimizer.start_automatic_optimization()
                    st.session_state.auto_optimizer_running = True
                    
                    # Log activity
                    st.session_state.optimization_activity.append({
                        "time": datetime.now(),
                        "action": "Started auto-optimization",
                        "details": f"Strategy: {auto_strategy}, Interval: {check_interval}s"
                    })
                    
                    st.success("Auto-optimization started!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting auto-optimization: {str(e)}")
        else:
            stop_button = st.button("Stop Auto-Optimization")
            if stop_button:
                try:
                    # Stop the background thread
                    st.session_state.auto_optimizer.stop_automatic_optimization()
                    st.session_state.auto_optimizer_running = False
                    
                    # Log activity
                    st.session_state.optimization_activity.append({
                        "time": datetime.now(),
                        "action": "Stopped auto-optimization",
                        "details": ""
                    })
                    
                    st.success("Auto-optimization stopped!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error stopping auto-optimization: {str(e)}")
    
    with control_col2:
        if st.button("Manual Check Now"):
            if st.session_state.monitored_prompts:
                with st.spinner("Checking prompts for optimization opportunities..."):
                    try:
                        # Create a temporary feedback collector and optimizer
                        temp_optimizer = optimizer
                        optimized_any = False
                        
                        # Check each prompt
                        for prompt_id in st.session_state.monitored_prompts.keys():
                            # Update last check time
                            st.session_state.monitored_prompts[prompt_id]["last_check"] = datetime.now()
                            
                            # Check if prompt is ready for optimization
                            readiness = temp_optimizer.get_optimization_stats(prompt_id)
                            
                            # Log the readiness check
                            st.session_state.optimization_activity.append({
                                "time": datetime.now(),
                                "action": "Checked optimization readiness",
                                "details": f"Prompt {prompt_id} - Ready: {readiness.get('is_ready', False)}, Feedback: {readiness.get('feedback_count', 0)}/{readiness.get('threshold', 5)}"
                            })
                            
                            if readiness["is_ready"]:
                                # Optimize the prompt
                                result = temp_optimizer.optimize_prompt(prompt_id)
                                
                                if result:
                                    # Update last optimized time
                                    st.session_state.monitored_prompts[prompt_id]["last_optimized"] = datetime.now()
                                    optimized_any = True
                                    
                                    # Log activity
                                    st.session_state.optimization_activity.append({
                                        "time": datetime.now(),
                                        "action": "Optimized prompt",
                                        "details": f"Prompt {prompt_id} optimized, new version: {result}"
                                    })
                            else:
                                # Log why it's not ready
                                if "strategy_assessment" in readiness:
                                    reason = readiness["strategy_assessment"].get("reason", "Unknown reason")
                                    st.session_state.optimization_activity.append({
                                        "time": datetime.now(),
                                        "action": "Not ready for optimization",
                                        "details": f"Reason: {reason}"
                                    })
                        
                        if optimized_any:
                            st.success("Successfully optimized one or more prompts!")
                        else:
                            st.info("No prompts were ready for optimization.")
                    except Exception as e:
                        st.error(f"Error during manual check: {str(e)}")
                        # Also log the error
                        st.session_state.optimization_activity.append({
                            "time": datetime.now(),
                            "action": "Error during check",
                            "details": str(e)
                        })
            else:
                st.warning("No prompts are being monitored. Add a prompt first.")
    
    # Activity log without dataframes
    st.subheader("Optimization Activity")
    
    if st.session_state.optimization_activity:
        # Display recent activity first
        for activity in reversed(st.session_state.optimization_activity):
            st.write(f"**{activity['time'].strftime('%H:%M:%S')}** - {activity['action']}: {activity['details']}")
            st.write("---")
    else:
        st.info("No optimization activity yet. Start the auto-optimizer or perform a manual check.")
    
    # Simulated feedback section for testing
    st.subheader("Simulate Feedback (for testing)")
    
    if st.session_state.monitored_prompts:
        feedback_prompt = st.selectbox(
            "Select prompt to add feedback to",
            options=list(st.session_state.monitored_prompts.keys())
        )
        
        feedback_positive = st.radio("Feedback type", ["Positive", "Negative"]) == "Positive"
        
        if st.button("Generate Test Feedback"):
            try:
                # Create a simulated response
                instance_id = optimizer.record_prompt_use(
                    prompt_id=feedback_prompt,
                    formatted_text=f"Simulated test for auto-optimizer at {datetime.now().strftime('%H:%M:%S')}"
                )
                
                response_id = optimizer.record_response(
                    prompt_instance_id=instance_id,
                    content=f"This is a simulated response for testing the auto-optimizer"
                )
                
                # Add feedback
                feedback_id = optimizer.record_feedback(
                    response_id=response_id,
                    is_positive=feedback_positive,
                    score=0.8 if feedback_positive else 0.2,
                    comments="Auto-generated test feedback"
                )
                
                # Log activity
                st.session_state.optimization_activity.append({
                    "time": datetime.now(),
                    "action": "Added test feedback",
                    "details": f"Added {'positive' if feedback_positive else 'negative'} feedback to prompt {feedback_prompt}"
                })
                
                st.success(f"Test feedback added! Feedback ID: {feedback_id}")
            except Exception as e:
                st.error(f"Error generating test feedback: {str(e)}")
    else:
        st.info("Add a prompt to monitoring first to generate test feedback.")

@st.cache_resource
def get_optimizer_with_openai(strategy_name="simple_ai"):
    """Get an optimizer instance with OpenAI integration if possible."""
    # Check if we have an API key in environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        try:
            # Create optimizer
            optimizer = PromptOptimizer(
                storage_dir="./streamlit_data",
                optimization_threshold=3,  # Lower threshold for demo
                strategy_name=strategy_name
            )
            
            # Create and set LLM service with the API key
            llm_service = LLMService(api_key=api_key)
            # Inject the LLM service into the strategy
            optimizer.optimizer.strategy.llm_service = llm_service
            
            return optimizer
        except Exception as e:
            st.warning(f"Error initializing OpenAI: {str(e)}")
            # Fallback to default optimizer
            return get_optimizer(strategy_name)
    else:
        # Use default optimizer without OpenAI
        return get_optimizer(strategy_name)

# Replace your optimizer initialization
try:
    # Try to get optimizer with OpenAI integration
    optimizer = get_optimizer_with_openai("simple_ai")
    if os.environ.get("OPENAI_API_KEY"):
        st.sidebar.success("✓ Using OpenAI integration from .env file")
    else:
        st.sidebar.info("ℹ️ Using simulated responses. Add OPENAI_API_KEY to .env file for real AI-powered optimizations.")
except Exception as e:
    st.sidebar.error(f"Failed to initialize OpenAI: {str(e)}")
    # Fallback to default optimizer
    optimizer = get_optimizer("simple_ai")

# Footer
st.markdown("---")
st.caption("PromptCraft API Demo Interface")