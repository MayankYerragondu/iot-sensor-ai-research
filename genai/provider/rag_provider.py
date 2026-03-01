import httpx
from app.providers.base import AiProvider
from app.schemas.chat import ChatRequest
from app.config import settings

class RagProvider(AiProvider):

    def type(self):
        return "RAG"

    async def chat(self, request: ChatRequest) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.RAG_SERVER_URL}/ask",
                json={
                    "tenant_id": request.tenant_id,
                    "device_id": request.device_id,
                    "messages": [m.dict() for m in request.messages]
                }
            )
        return resp.json()["answer"]