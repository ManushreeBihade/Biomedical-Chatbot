from fastapi import APIRouter, HTTPException
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.llm_service import generate_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = generate_response(
            provider=request.provider,
            prompt=request.prompt,
            memory_enabled=request.memory_enabled,
            history=request.history
        )
        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
