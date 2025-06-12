from fastapi import FastAPI
from prompt_optimizer.api import PromptOptimizer

app = FastAPI(title="PromptCraft API")
optimizer = PromptOptimizer()

@app.post("/prompts")
async def register_prompt(text: str, description: str = ""):
    result = optimizer.register_prompt(text, description)
    return result.model_dump()

@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str):
    result = optimizer.get_prompt(prompt_id)
    return result.model_dump()

@app.post("/prompts/{prompt_id}/use")
async def record_prompt_use(prompt_id: str, formatted_text: str):
    result = optimizer.record_prompt_use(prompt_id, formatted_text)
    return result.model_dump()

@app.post("/responses/{prompt_instance_id}")
async def record_response(prompt_instance_id: str, content: str):
    result = optimizer.record_response(prompt_instance_id, content)
    return result.model_dump()

@app.post("/feedback/{response_id}")
async def record_feedback(response_id: str, score: float):
    result = optimizer.record_feedback(response_id, score)
    return result.model_dump()

@app.post("/optimize/{prompt_id}")
async def optimize_prompt(prompt_id: str, force: bool = False):
    result = optimizer.optimize_prompt(prompt_id, force)
    return result.model_dump()

@app.get("/optimization/{prompt_id}")
async def optimization_stats(prompt_id: str):
    result = optimizer.get_optimization_stats(prompt_id)
    return result.model_dump()

@app.get("/config")
async def get_config():
    result = optimizer.get_config_info()
    return result.model_dump()
