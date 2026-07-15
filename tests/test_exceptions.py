"""Tests for MVT 1.2 — Exception hierarchy."""
import pytest
from src.agent.exceptions import (
    AgentError,
    ConfigError,
    LLMError,
    LLMTimeoutError,
    LLMAuthError,
    LLMRateLimitError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    SessionError,
    SessionNotFoundError,
    SessionSaveError,
    CompressionError,
    LoopError,
)


class TestExceptionHierarchy:
    """验证继承链正确性"""

    def test_all_exceptions_are_agent_errors(self):
        exceptions = [
            ConfigError,
            LLMError,
            LLMTimeoutError,
            LLMAuthError,
            LLMRateLimitError,
            ToolError,
            ToolNotFoundError,
            ToolExecutionError,
            SessionError,
            SessionNotFoundError,
            SessionSaveError,
            CompressionError,
            LoopError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, AgentError), f"{exc_cls.__name__} 应继承 AgentError"

    def test_llm_subclasses(self):
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMAuthError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_tool_subclasses(self):
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)

    def test_session_subclasses(self):
        assert issubclass(SessionNotFoundError, SessionError)
        assert issubclass(SessionSaveError, SessionError)


class TestExceptionInstances:
    """验证实例化和 isinstance 检查"""

    def test_instance_isinstance_chain(self):
        exc = LLMAuthError("Invalid API key")
        assert isinstance(exc, LLMAuthError)
        assert isinstance(exc, LLMError)
        assert isinstance(exc, AgentError)
        assert isinstance(exc, Exception)

    def test_error_message(self):
        exc = ConfigError("Missing DEEPSEEK_API_KEY")
        assert "DEEPSEEK_API_KEY" in str(exc)
