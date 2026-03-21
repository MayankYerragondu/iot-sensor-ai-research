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

# This file defines the schema for the chat request and response. It uses Pydantic to validate the data and ensure that it is in the correct format. The ChatRequest class defines the structure of the request, which includes the tenant_id, device_id, mode, and messages. The ChatResponse class defines the structure of the response, which includes the provider and answer.
