from fastapi import APIRouter, Depends
from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer

router = APIRouter()

@router.get("/", summary="Get configuration")
def get_config(optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.get_config_info()
