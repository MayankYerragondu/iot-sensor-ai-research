from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
service = ChatService()

# Endpoint for chat interactions
@router.post("/genai-chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await service.chat(request)