"""Tests for MVT 2.1 — Token counting."""
import pytest
from unittest.mock import patch, MagicMock

from src.agent.models import Message, ToolCall, ToolCallFunction
from src.agent.tokens import count_tokens, estimate_tokens_from_text


class TestCountTokens:
    """Token 计数测试"""

    def test_count_single_message(self):
        messages = [Message(role="user", content="Hello, world!")]
        count = count_tokens(messages, "deepseek-chat")
        assert count > 0
        assert isinstance(count, int)

    def test_count_multiple_messages(self):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What is 1+1?"),
            Message(role="assistant", content="It's 2."),
        ]
        count = count_tokens(messages, "deepseek-chat")
        assert count > 0
        # 多条消息 token 应大于单条
        single = count_tokens([messages[0]], "deepseek-chat")
        assert count > single

    def test_count_with_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                content="Let me calculate.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=ToolCallFunction(name="calculator", arguments='{"expression":"1+1"}'),
                    )
                ],
            ),
            Message(role="tool", content="2", tool_call_id="c1", name="calculator"),
        ]
        count = count_tokens(messages, "deepseek-chat")
        assert count > 0
        # 含 tool_calls 的消息 token 应多于纯文本
        plain = count_tokens([Message(role="user", content="Hi")], "deepseek-chat")
        assert count > plain

    def test_fallback_without_tiktoken(self):
        """测试无 tiktoken 时的降级"""
        with patch("src.agent.tokens._TIKTOKEN_AVAILABLE", False):
            messages = [Message(role="user", content="Hello, world!")]
            count = count_tokens(messages, "deepseek-chat")
            assert count > 0


class TestEstimateTokens:
    """文本 token 估算测试"""

    def test_short_text(self):
        tokens = estimate_tokens_from_text("Hello")
        assert tokens >= 1

    def test_long_text(self):
        text = "This is a longer piece of text that should have more tokens."
        tokens = estimate_tokens_from_text(text)
        # 大致 token 数 < 字符数 / 4
        assert tokens <= len(text)
        assert tokens >= 1

    def test_empty_text(self):
        tokens = estimate_tokens_from_text("")
        assert tokens == 1  # 至少 1
