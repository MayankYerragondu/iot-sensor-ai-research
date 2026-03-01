from openai import AsyncOpenAI
from app.providers.base import AiProvider
from app.schemas.chat import ChatRequest
from app.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# OpenAIProvider implements the AiProvider interface to interact with OpenAI's API for chat completions.
class OpenAIProvider(AiProvider):

    def type(self):
        return "OPENAI"

    #   The chat method sends a request to OpenAI's chat completions API with the specified model and messages from the ChatRequest. It then returns the content of the first message in the response choices.
    async def chat(self, request: ChatRequest) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[m.dict() for m in request.messages]
        )
        return response.choices[0].message.content