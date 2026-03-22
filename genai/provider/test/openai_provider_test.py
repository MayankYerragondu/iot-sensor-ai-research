import pytest
from unittest.mock import AsyncMock, patch

from app.providers.openai_provider import OpenAIProvider
from app.schemas.chat import ChatRequest


# ------------------------
# Fixtures
# ------------------------

@pytest.fixture
def provider():
    return OpenAIProvider()


@pytest.fixture
def chat_request():
    # Adjust fields based on your actual ChatRequest schema
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content

        def dict(self):
            return {
                "role": self.role,
                "content": self.content
            }

    return ChatRequest(
        messages=[
            Message("user", "Hello"),
            Message("assistant", "Hi there")
        ]
    )


# ------------------------
# Tests
# ------------------------

def test_type(provider):
    assert provider.type() == "OPENAI"


def test_name(provider):
    assert provider.name() == "OpenAI Provider"


@pytest.mark.asyncio
@patch("app.providers.openai_provider.client")
async def test_chat(mock_client, provider, chat_request):
    # ------------------------
    # Mock OpenAI response
    # ------------------------
    mock_response = AsyncMock()
    mock_choice = AsyncMock()
    mock_message = AsyncMock()

    mock_message.content = "Mocked response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    # Mock the async API call
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # ------------------------
    # Execute
    # ------------------------
    result = await provider.chat(chat_request)

    # ------------------------
    # Assertions
    # ------------------------
    assert result == "Mocked response"

    mock_client.chat.completions.create.assert_awaited_once()

    # Validate payload transformation
    called_args = mock_client.chat.completions.create.call_args.kwargs

    assert called_args["model"] == "gpt-4o-mini"
    assert called_args["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]