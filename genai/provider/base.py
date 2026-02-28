from abc import ABC, abstractmethod
from app.schemas.chat import ChatRequest

# AiProvider is an abstract base class that defines the interface for AI providers. It requires implementing classes to specify their type and how they handle chat requests.
class AiProvider(ABC):


    @abstractmethod
    def type(self) -> str:
        pass

    # The chat method is an asynchronous abstract method that takes a ChatRequest and returns a string response. Implementing classes must provide their own logic for handling chat interactions.
    @abstractmethod
    async def chat(self, request: ChatRequest) -> str:
        pass

    @abstractmethod
    async def stream_chat(self, request: ChatRequest):
        stream = await self.chat(request)
        stream = stream.split("\n")
        for line in stream:
            yield line
        pass