from app.schemas.chat import ChatRequest, ChatResponse
from app.services.router import ProviderRouter

class ChatService:

    def __init__(self):
        self.router = ProviderRouter()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        provider = self.router.choose(request.mode)
        answer = await provider.chat(request)

        return ChatResponse(
            provider=provider.type(),
            answer=answer
        )