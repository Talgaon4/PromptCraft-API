from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer

router = APIRouter()

class FeedbackCreate(BaseModel):
    response_id: str
    score: float

@router.post("/feedback", summary="Record feedback")
def record_feedback(data: FeedbackCreate, optimizer: PromptOptimizer = Depends(get_optimizer)):
    result = optimizer.record_feedback(response_id=data.response_id, score=data.score)
    if result.success:
        return result
    raise HTTPException(status_code=400, detail=result.message)
