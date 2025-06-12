from fastapi import FastAPI, Depends
from prompt_optimizer.api import PromptOptimizer

app = FastAPI(title="PromptCraft API")

# Store the optimizer instance on the application state so tests can
# override it via dependency injection.
app.state.optimizer = PromptOptimizer()


def get_optimizer() -> PromptOptimizer:
    """Retrieve the current optimizer instance."""
    return app.state.optimizer

@app.post("/prompts")
async def register_prompt(
    text: str,
    description: str = "",
    optimizer: PromptOptimizer = Depends(get_optimizer),
):
    result = optimizer.register_prompt(text, description)
    return result.model_dump()

@app.get("/prompts/{prompt_id}")
async def get_prompt(
    prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)
):
    result = optimizer.get_prompt(prompt_id)
    return result.model_dump()

@app.post("/prompts/{prompt_id}/use")
async def record_prompt_use(
    prompt_id: str,
    formatted_text: str,
    optimizer: PromptOptimizer = Depends(get_optimizer),
):
    result = optimizer.record_prompt_use(prompt_id, formatted_text)
    return result.model_dump()

@app.post("/responses/{prompt_instance_id}")
async def record_response(
    prompt_instance_id: str,
    content: str,
    optimizer: PromptOptimizer = Depends(get_optimizer),
):
    result = optimizer.record_response(prompt_instance_id, content)
    return result.model_dump()

@app.post("/feedback/{response_id}")
async def record_feedback(
    response_id: str,
    score: float,
    optimizer: PromptOptimizer = Depends(get_optimizer),
):
    result = optimizer.record_feedback(response_id, score)
    return result.model_dump()

@app.post("/optimize/{prompt_id}")
async def optimize_prompt(
    prompt_id: str,
    force: bool = False,
    optimizer: PromptOptimizer = Depends(get_optimizer),
):
    result = optimizer.optimize_prompt(prompt_id, force)
    return result.model_dump()

@app.get("/optimization/{prompt_id}")
async def optimization_stats(
    prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)
):
    result = optimizer.get_optimization_stats(prompt_id)
    return result.model_dump()

@app.get("/config")
async def get_config(optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.get_config_info()
    return result.model_dump()
