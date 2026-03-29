import pytest
import asyncio
from app.providers.ai_provider import AiProvider
from app.schemas.chat import ChatRequest


# Mock implementation of AiProvider for testing
class MockAiProvider(AiProvider):

    def type(self) -> str:
        return "mock"

    async def chat(self, request: ChatRequest) -> str:
        return "line1\nline2\nline3"

    async def stream_chat(self, request: ChatRequest):
        # use base logic intentionally
        async for line in super().stream_chat(request):
            yield line

    def name(self) -> str:
        return "Mock Provider"


# Dummy ChatRequest (adjust fields based on your actual schema)
@pytest.fixture
def chat_request():
    return ChatRequest(
        message="hello",
        user_id="test-user"
    )


# Provider fixture
@pytest.fixture
def provider():
    return MockAiProvider()


# ------------------------
# Tests
# ------------------------

# Test the type method
def test_type(provider):
    assert provider.type() == "mock"


# Test the name method
def test_name(provider):
    assert provider.name() == "Mock Provider"


# Chat tests
@pytest.mark.asyncio
async def test_chat(provider, chat_request):
    response = await provider.chat(chat_request)
    assert response == "line1\nline2\nline3"

# Stream chat tests
@pytest.mark.asyncio
async def test_stream_chat(provider, chat_request):
    result = []
    async for line in provider.stream_chat(chat_request):
        result.append(line)

    assert result == ["line1", "line2", "line3"]