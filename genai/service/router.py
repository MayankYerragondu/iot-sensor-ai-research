from app.providers.openai_provider import OpenAIProvider
from app.providers.rag_provider import RagProvider

class ProviderRouter:

    def __init__(self):
        self.openai = OpenAIProvider()
        self.rag = RagProvider()

    def choose(self, mode: str):
        if mode == "OPENAI":
            return self.openai
        if mode == "RAG":
            return self.rag
        return self.rag  
    # AUTO default to RAG for IoT