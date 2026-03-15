import os

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag-service:8001")

settings = Settings()