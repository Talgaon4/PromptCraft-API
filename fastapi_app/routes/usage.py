from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import OperationResult

router = APIRouter(tags=["usage"])


class UsePayload(BaseModel):
    formatted_text: str
    context: Optional[Dict[str, Any]] = None


@router.post("/prompts/{prompt_id}/use", response_model=OperationResult)
def record_prompt_use(prompt_id: str, payload: UsePayload, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.record_prompt_use(prompt_id=prompt_id, formatted_text=payload.formatted_text, context=payload.context)


class ResponsePayload(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("/instances/{instance_id}/responses", response_model=OperationResult)
def record_response(instance_id: str, payload: ResponsePayload, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.record_response(prompt_instance_id=instance_id, content=payload.content, metadata=payload.metadata)
