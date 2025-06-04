from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer

router = APIRouter()

class PromptUseRequest(BaseModel):
    formatted_text: str
    context: Optional[dict] = None

class ResponseCreate(BaseModel):
    prompt_instance_id: str
    content: str
    metadata: Optional[dict] = None

@router.post("/prompts/{prompt_id}/instances", summary="Record prompt use")
def record_use(prompt_id: str, data: PromptUseRequest, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.record_prompt_use(prompt_id=prompt_id, formatted_text=data.formatted_text, context=data.context)
    if result.success:
        return result
    raise HTTPException(status_code=400, detail=result.message)

@router.post("/responses", summary="Record a response")
def record_response(data: ResponseCreate, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.record_response(prompt_instance_id=data.prompt_instance_id, content=data.content, metadata=data.metadata)
    if result.success:
        return result
    raise HTTPException(status_code=400, detail=result.message)
