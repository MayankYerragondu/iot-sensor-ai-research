from openai import AsyncOpenAI
from app.providers.base import AiProvider
from app.schemas.chat import ChatRequest
from app.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class OpenAIProvider(AiProvider):

    def type(self):
        return "OPENAI"

    async def chat(self, request: ChatRequest) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[m.dict() for m in request.messages]
        )
        return response.choices[0].message.content