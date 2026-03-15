from app.schemas.chat import ChatRequest, ChatResponse
from app.services.router import ProviderRouter

# This class is responsible for handling the chat requests and routing them to the appropriate provider
class ChatService:

    # Initialize the provider router
    def __init__(self):
        self.router = ProviderRouter()

    # Handle the chat request and route it to the appropriate provider
    async def chat(self, request: ChatRequest) -> ChatResponse:
        provider = self.router.choose(request.mode)
        answer = await provider.chat(request)

        # Return the response from the provider
        return ChatResponse(
            provider=provider.type(),
            answer=answer
        )
    
    # AUTO default to RAG for IoT
    async def chat(self, request: ChatRequest) -> ChatResponse:
        provider = self.router.choose(request.mode)
        answer = await provider.chat(request)

        # Return the response from the provider
        return ChatResponse(
            provider=provider.type(),
            answer=answer
        )
    