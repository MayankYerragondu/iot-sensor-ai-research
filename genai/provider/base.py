from abc import ABC, abstractmethod
from app.schemas.chat import ChatRequest

class AiProvider(ABC):

    @abstractmethod
    def type(self) -> str:
        pass

    @abstractmethod
    async def chat(self, request: ChatRequest) -> str:
        pass