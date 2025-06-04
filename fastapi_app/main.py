from fastapi import FastAPI
from .routes import prompts, responses, feedback, optimization, config

app = FastAPI(title="PromptCraft API")

app.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
app.include_router(responses.router, tags=["responses"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(optimization.router, tags=["optimization"])
app.include_router(config.router, prefix="/config", tags=["config"])
