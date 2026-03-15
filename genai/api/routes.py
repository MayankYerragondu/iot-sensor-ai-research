from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()
service = ChatService()

# Endpoint for chat interactions
@router.post("/genai-chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await service.chat(request)

# This file defines the API routes for the chat interactions. It uses FastAPI to create an endpoint at "/genai-chat" that accepts POST requests with a ChatRequest body and returns a ChatResponse. The chat function calls the chat method of the ChatService to handle the request and generate the response.
