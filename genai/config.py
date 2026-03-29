import os

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag-service:8001")

# Global settings instance
settings = Settings()

# Export the settings instance for use in other modules
__all__ = ["settings"]
