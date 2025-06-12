# PromptCraft API

A Python API for automatically optimizing AI prompts based on user feedback.

## Features

- Track prompts and their usage
- Collect and analyze user feedback on AI responses
- Automatically generate improved prompts
- Version control for prompts
- Interactive testing interface

## Installation

```bash
# Clone the repository
git clone https://github.com/talgaon4/PromptCraft-API.git
cd PromptCraft-API

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from prompt_optimizer.api import PromptOptimizer

# Initialize the optimizer
optimizer = PromptOptimizer()

# Register a prompt
prompt_id = optimizer.register_prompt(
    text="Summarize the following text: {input_text}",
    description="Text summarization prompt"
)

# Record usage and response
instance_id = optimizer.record_prompt_use(
    prompt_id=prompt_id,
    formatted_text="Summarize the following text: This is a test."
)

response_id = optimizer.record_response(
    prompt_instance_id=instance_id,
    content="This is a test summary."
)

# Record feedback
optimizer.record_feedback(
    response_id=response_id,
    score=0.8,
)
# Generate an optimization
optimizer.optimize_prompt(prompt_id)
```

## Interactive Interface

To launch the Streamlit interactive testing environment:

```bash
streamlit run interface/app.py
```

The interface allows you to:
- Create and test prompts with real-time feedback
- Visualize prompt performance
- Generate optimized versions automatically
- Track version history

## FastAPI Application

To run the HTTP API with FastAPI:

```bash
uvicorn fastapi_app.main:app --reload
```

This server exposes endpoints for registering prompts, recording usage,
submitting feedback and triggering optimizations.

## License

MIT License
