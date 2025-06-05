from fastapi import FastAPI

from .routes import prompts, usage, feedback, optimization, config

app = FastAPI(title="PromptCraft API")

app.include_router(prompts.router)
app.include_router(usage.router)
app.include_router(feedback.router)
app.include_router(optimization.router)
app.include_router(config.router)
