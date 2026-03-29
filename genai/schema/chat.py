from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    tenant_id: Optional[str]
    device_id: Optional[str]
    mode: str = "AUTO"  # OPENAI | RAG | AUTO
    messages: List[Message]

class ChatResponse(BaseModel):
    provider: str
    answer: str

