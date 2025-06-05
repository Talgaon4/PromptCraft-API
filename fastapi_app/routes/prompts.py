from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Any

from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import PromptResult, OperationResult


class PromptCreate(BaseModel):
    text: str
    description: str = ""


router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("/", response_model=PromptResult)
def register_prompt(payload: PromptCreate, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.register_prompt(text=payload.text, description=payload.description)


@router.get("/{prompt_id}", response_model=PromptResult)
def get_prompt(prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.get_prompt(prompt_id)


@router.get("/{prompt_id}/history", response_model=OperationResult)
def get_prompt_history(prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)):
    try:
        history = optimizer.optimizer.get_optimization_history(prompt_id)
        return OperationResult(is_successful=True, data=history, message="History retrieved")
    except Exception as e:
        return OperationResult(is_successful=False, message="Failed to retrieve history", errors=[str(e)])
