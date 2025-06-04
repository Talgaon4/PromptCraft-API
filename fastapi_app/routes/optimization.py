from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer

router = APIRouter()

class OptimizeRequest(BaseModel):
    force: bool = False

@router.get("/prompts/{prompt_id}/stats", summary="Get optimization stats")
def get_stats(prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.get_optimization_stats(prompt_id)
    if result.success:
        return result
    raise HTTPException(status_code=404, detail=result.message)

@router.post("/prompts/{prompt_id}/optimize", summary="Optimize a prompt")
def optimize(prompt_id: str, data: OptimizeRequest, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.optimize_prompt(prompt_id, force=data.force)
    if result.success:
        return result
    return result
