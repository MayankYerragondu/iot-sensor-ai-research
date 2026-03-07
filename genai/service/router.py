from app.providers.openai_provider import OpenAIProvider
from app.providers.rag_provider import RagProvider

# This class is responsible for routing the requests to the appropriate provider based on the mode
class ProviderRouter:

    # Initialize the providers
    def __init__(self):
        # Initialize the providers
        self.openai = OpenAIProvider()
        self.rag = RagProvider()

    # Choose the provider based on the mode
    def choose(self, mode: str):
        # Choose the provider based on the mode
        if mode == "OPENAI":
            return self.openai
        if mode == "RAG":
            return self.rag
        return self.rag  
    # AUTO default to RAG for IoT