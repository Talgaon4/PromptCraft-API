from fastapi import APIRouter, Depends

from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import OperationResult

router = APIRouter(tags=["config"])


@router.get("/config", response_model=OperationResult)
def get_config(optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.get_config_info()
