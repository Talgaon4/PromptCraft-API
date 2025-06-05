from fastapi import APIRouter, Depends, Query

from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import OptimizationResult

router = APIRouter(tags=["optimization"])


@router.get("/prompts/{prompt_id}/stats", response_model=OptimizationResult)
def get_stats(prompt_id: str, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.get_optimization_stats(prompt_id)


@router.post("/prompts/{prompt_id}/optimize", response_model=OptimizationResult)
def optimize_prompt(prompt_id: str, force: bool = Query(False), optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.optimize_prompt(prompt_id, force=force)
