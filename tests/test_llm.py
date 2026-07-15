"""Tests for MVT 1.4 — LLM Client."""
import inspect
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AuthenticationError, RateLimitError, APITimeoutError

from src.agent.config import Config
from src.agent.exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.agent.llm.base import LLMClient
from src.agent.llm.deepseek import DeepSeekClient
from src.agent.models import LLMResponse, Message, Usage


def _make_mock_response(status_code: int = 200) -> httpx.Response:
    """构造 mock httpx.Response 用于 OpenAI 异常"""
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


@pytest.fixture
def config():
    return Config(
        llm_api_key="sk-test",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com/v1",
        llm_timeout=10,
        llm_max_retries=1,
    )


@pytest.fixture
def client(config):
    return DeepSeekClient(config)


class TestLLMClientInterface:
    """验证 LLMClient 接口完整性"""

    def test_deepseek_client_is_llm_client(self, client):
        assert isinstance(client, LLMClient)

    def test_chat_is_async_callable(self, client):
        assert callable(client.chat)
        assert inspect.iscoroutinefunction(client.chat)


class TestDeepSeekClient:
    """DeepSeek 客户端单元测试（Mock）"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, client):
        """Mock：简单文本回复"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, how can I help?"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.model = "deepseek-chat"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch.object(client._client.chat.completions, "create",
                          AsyncMock(return_value=mock_response)):
            result = await client.chat([Message(role="user", content="Hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, how can I help?"
        assert result.finish_reason == "stop"
        assert result.tool_calls is None
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_auth_error(self, client):
        """Mock：鉴权失败"""
        mock_resp = _make_mock_response(401)
        with patch.object(client._client.chat.completions, "create",
                          AsyncMock(side_effect=AuthenticationError("Invalid API key", response=mock_resp, body=None))):
            with pytest.raises(LLMAuthError):
                await client.chat([Message(role="user", content="Hi")])

    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """Mock：超时"""
        with patch.object(client._client.chat.completions, "create",
                          AsyncMock(side_effect=APITimeoutError("timeout"))):
            with pytest.raises(LLMTimeoutError):
                await client.chat([Message(role="user", content="Hi")])

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, client):
        """Mock：限流"""
        mock_resp = _make_mock_response(429)
        with patch.object(client._client.chat.completions, "create",
                          AsyncMock(side_effect=RateLimitError("Rate limited", response=mock_resp, body=None))):
            with pytest.raises(LLMRateLimitError):
                await client.chat([Message(role="user", content="Hi")])

    @pytest.mark.asyncio
    async def test_tool_call_response(self, client):
        """Mock：Tool Call 响应"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.type = "function"
        mock_tc.function.name = "calculator"
        mock_tc.function.arguments = '{"expression": "1+1"}'
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"
        mock_response.choices = [mock_choice]
        mock_response.model = "deepseek-chat"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70

        with patch.object(client._client.chat.completions, "create",
                          AsyncMock(return_value=mock_response)):
            result = await client.chat([Message(role="user", content="1+1=?")])

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "calculator"


class TestIntegrationLLM:
    """集成测试（需要 API Key）"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_chat(self):
        """真实 LLM 调用：简单对话"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")

        config = Config.from_env(env_file=None)
        client = DeepSeekClient(config)
        result = await client.chat([Message(role="user", content="Hello! Say 'hi' back.")])

        assert result.content is not None
        assert isinstance(result.usage, Usage)
        assert result.usage.total_tokens > 0
        assert result.model is not None
