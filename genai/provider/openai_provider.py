from openai import AsyncOpenAI
from app.providers.base import AiProvider
from app.schemas.chat import ChatRequest
from app.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# OpenAIProvider implements the AiProvider interface to interact with OpenAI's API for chat completions.
class OpenAIProvider(AiProvider):

    def type(self):
        return "OPENAI"

    # The name method returns a dictionary with the provider's name and additional metadata.
    def name(self):
        """
        Returns metadata about this AI provider implementation.
        This can be used for display purposes in UIs or logs.
        """
        return {
            "name": "OpenAI Provider",
            "type": self.type(),
            "model": "gpt-4o-mini"
        }