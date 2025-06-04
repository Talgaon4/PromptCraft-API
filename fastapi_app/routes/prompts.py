from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer

class PromptCreate(BaseModel):
    text: str
    description: str = ""

router = APIRouter()

@router.post("/", summary="Create a new prompt")
def create_prompt(data: PromptCreate, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.register_prompt(text=data.text, description=data.description)
    if result.success:
        return result
    raise HTTPException(status_code=400, detail=result.message)

@router.get("/{prompt_id}", summary="Get a prompt")
def get_prompt(prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.get_prompt(prompt_id)
    if result.success:
        return result
    raise HTTPException(status_code=404, detail=result.message)

@router.get("/", summary="List prompts")
def list_prompts(optimizer: PromptOptimizer = Depends(get_optimizer)):
    prompts = optimizer.prompt_manager.list_prompts()
    return [p.model_dump() for p in prompts]
