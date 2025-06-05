from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..deps import get_optimizer
from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import OperationResult

router = APIRouter(tags=["feedback"])


class FeedbackPayload(BaseModel):
    score: float


@router.post("/responses/{response_id}/feedback", response_model=OperationResult)
def record_feedback(response_id: str, payload: FeedbackPayload, optimizer: PromptOptimizer = Depends(get_optimizer)):
    return optimizer.record_feedback(
        response_id=response_id,
        score=payload.score,
    )
