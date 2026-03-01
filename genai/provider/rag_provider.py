import httpx
from app.providers.base import AiProvider
from app.schemas.chat import ChatRequest
from app.config import settings

# RagProvider implements the AiProvider interface to interact with a RAG (Retrieval-Augmented Generation) server for chat completions. It defines the type of provider and how to handle chat requests by sending them to the RAG server and returning the response.
class RagProvider(AiProvider):

    def type(self):
        return "RAG"

    # The chat method sends a POST request to the RAG server with the tenant ID, device ID, and messages from the ChatRequest. It then returns the answer from the RAG server's response.
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